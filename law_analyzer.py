import os
import re
import json
import html
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import pyLDAvis.gensim_models
import textstat
import traceback
from draft_importer import DraftImporter
from loophole import LoopholeAnalyzer

_original_default = json.JSONEncoder.default

def _patched_default(self, obj):
    if isinstance(obj, np.complexfloating):
        return float(obj.real)
    return _original_default(self, obj)

json.JSONEncoder.default = _patched_default

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.complexfloating):
            return float(obj.real)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class LawDraftSystem:
    """Main analysis system with file outputs"""
    
    def __init__(self, data_dir: str = "draft_data", report_dir: str = "analysis_reports"):
        self.data_dir = Path(data_dir)
        self.report_dir = Path(report_dir)
        self.drafts = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer()
        self.importer = DraftImporter()
        self.loophole_analyzer = LoopholeAnalyzer()
        
        # Add these verification steps
        print(f"Initializing directories:\n- Data: {data_dir}\n- Reports: {report_dir}")

        self.data_dir = Path(data_dir)
        self.report_dir = Path(report_dir)

        # Verify directory creation
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.report_dir.mkdir(exist_ok=True, parents=True)
        data_count = len(list(self.data_dir.glob('*')))
        report_count = len(list(self.report_dir.glob('*')))
        print(f"Directories confirmed:\n- Data directory contains {data_count} outputs\n- Report directory contains {report_count} outputs")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        if isinstance(obj, np.integer):
               return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.complexfloating):
            return float(obj.real)  # Convert complex to float by taking real part
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def import_draft(self, filepath: str) -> str:
        """Import and analyze a draft with file outputs"""
        if not isinstance(filepath, str):
            raise ValueError("Filepath must be a string")
        
        print(f"Importing draft from: {filepath}")
        raw_data = self.importer.import_draft(filepath)
        print(f"Draft imported with keys: {list(raw_data.keys())}")
        if 'articles' not in raw_data or not raw_data['articles']:
            print("Warning: No articles found in draft data")
        draft_id = f"DRAFT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save raw data
        self.drafts[draft_id] = raw_data
        self._save_draft_data(draft_id)
        
        # Generate outputs
        self._save_analysis_reports(draft_id)
        return draft_id

    def _save_draft_data(self, draft_id: str):
        """Save raw draft data with type conversion"""
        data = self._convert_numpy_types(self.drafts[draft_id])
        with open(self.data_dir / f"{draft_id}.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_analysis_reports(self, draft_id: str) -> Dict[str, str]:
      """Generate all report outputs with the tabbed HTML as primary"""
      draft = self.drafts[draft_id]
      report_dir = self.report_dir / draft_id
      report_dir.mkdir(exist_ok=True)
          # Generate the main HTML report
      html_report = self._generate_analysis_html(draft_id)
          # Keep existing reports for compatibility
      json_path = report_dir / "analysis.json"
      with open(json_path, 'w') as f:
          json.dump({
              'metadata': draft.get('metadata', {}),
              'stats': self._calculate_stats(draft),
              'readability': self._calculate_readability(draft),
              'topics': self._analyze_topics(draft_id),
              'loopholes': self.loophole_analyzer.analyze_draft(draft)
          }, f, indent=2, cls=NumpyEncoder)
          return {
          'html': html_report,
          'json': str(json_path),
          'text': str(report_dir / "report.txt")  # Keep if you still generate this
      }

    def _calculate_stats(self, draft: Dict) -> Dict:
        """Calculate document statistics"""
        return {
            'article_count': len(draft['articles']),
            'word_count': len(draft['content'].split()),
            'line_count': len(draft['content'].splitlines())
        }

    def _calculate_readability(self, draft: Dict) -> Dict:
        """Calculate readability metrics with native floats"""
        return {
            'flesch': float(textstat.flesch_reading_ease(draft['content'])),
            'smog': float(textstat.smog_index(draft['content'])),
            'reading_time': f"{len(draft['content'].split())/200:.1f} minutes"
        }

    def _is_grouping_title(self, title: str) -> bool:
        """Detect if a title is a grouping like 'part 1', 'chapter 1', 'section 1' etc."""
        # Normalize title: strip, replace multiple whitespace with single space, lowercase, remove trailing punctuation
        title_clean = re.sub(r'\s+', ' ', title.strip()).lower().rstrip('.:;')
        # Match grouping words optionally followed by number or roman numerals or spelled out numbers
        pattern = re.compile(
            r'^(part|chapter|section|title|subtitle|division|book)(\s+((\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)))?$', 
            re.IGNORECASE
        )
        # Check full match
        full_match = bool(pattern.match(title_clean))
        # Check if short title (<=5 words) starts with grouping word + number
        words = title_clean.split()
        short_start_match = False
        if len(words) <= 5:
            short_start_match = bool(re.match(
                r'^(part|chapter|section|title|subtitle|division|book)(\s+((\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)))?', 
                title_clean, re.IGNORECASE))
        match = full_match or short_start_match
        print(f"DEBUG _is_grouping_title: title='{title}' cleaned='{title_clean}' full_match={full_match} short_start_match={short_start_match} match={match}")
        return match

    def _flatten_article_content(self, content):
        """Recursively flatten article content if nested dicts exist"""
        if isinstance(content, dict):
            texts = []
            for key, val in content.items():
                texts.append(self._flatten_article_content(val))
            return "\n".join(texts)
        elif isinstance(content, str):
            return content
        else:
            return str(content)

    def _analyze_topics(self, draft_id: str) -> List[Dict]:
        """Ensure each article becomes a topic, excluding grouping titles"""
        draft = self.drafts[draft_id]
        all_titles = list(draft['articles'].keys())
        print(f"DEBUG _analyze_topics: All article titles before filtering: {all_titles}")
        articles = [(title, content) for title, content in draft['articles'].items() if not self._is_grouping_title(title)]

        print(f"DEBUG _analyze_topics: Number of articles for topic analysis: {len(articles)}")
        print(f"DEBUG _analyze_topics: Article titles: {[title for title, _ in articles]}")

        # If there are no articles, return empty list
        if not articles:
            return []

        # Process each article's content
        processed_articles = [self._preprocess_text(self._flatten_article_content(content)) if isinstance(content, dict) else self._preprocess_text(content) for _, content in articles]
        article_titles = [title for title, _ in articles]

        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_articles)
        corpus = [dictionary.doc2bow(text) for text in processed_articles]

        # Adapt number of topics to number of articles
        num_topics = len(articles)
        print(f"DEBUG _analyze_topics: Number of topics set to: {num_topics}")

        # Train LDA model - one topic per article
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=15,
            random_state=42,
            alpha='auto'
        )

        # Create direct mapping of articles to topics
        topics = []
        for topic_id, ((article_title, _), processed_article, bow) in enumerate(zip(articles, processed_articles, corpus)):
            # Get top keywords for this topic, excluding single-letter words
            keywords = [word for word, _ in lda.show_topic(topic_id) if len(word) > 1]
            
            # Compute topic coherence for the article
            topic_coherence = max([score for _, score in lda.get_document_topics(bow)], default=0.0)

            topics.append({
                'topic_id': topic_id,
                'article': article_title,
                'article_score': float(topic_coherence),
                'keywords': keywords,
                'raw_content': self.drafts[draft_id]['articles'][article_title]  # Include raw article content
            })

        return topics

    def _analyze_articles(self, draft: Dict) -> Dict:
        """Analyze individual articles with enhanced metrics"""
        article_analysis = {}
        for header, content in draft['articles'].items():
            doc = self.nlp(content)
            
            # Extract key entities
            entities = list(set([ent.text for ent in doc.ents if ent.label_ in ['LAW', 'ORG', 'PERSON']]))
            
            # Extract important noun phrases
            noun_phrases = list(set([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]))
            
            article_analysis[header] = {
                'word_count': len(content.split()),
                'sentence_count': len(list(doc.sents)),
                'readability': textstat.flesch_reading_ease(content),
                'key_entities': entities[:5],  # Top 5 entities
                'key_phrases': noun_phrases[:5]  # Top 5 noun phrases
            }
        return article_analysis

    def _generate_topic_visualization(self, draft_id: str) -> str:
        """Generate complete analysis visualization with topics and loopholes"""
        try:
            print(f"\nStarting visualization for {draft_id}")

            # 1. Verify draft exists
            if draft_id not in self.drafts:
                print(f"Error: Draft {draft_id} not loaded")
                return ""

            draft = self.drafts[draft_id]

            # 2. Verify articles exist
            articles = list(draft['articles'].items())
            if not articles:
                print("No articles found in draft")
                return ""

            print(f"Processing {len(articles)} articles...")

            # 3. Create output directory (verified)
            output_dir = self.report_dir / draft_id
            output_dir.mkdir(exist_ok=True, parents=True)
            print(f"Output directory: {output_dir}")

            # 4. Generate LDA visualization (with verification)
            processed_articles = [self._preprocess_text(self._flatten_article_content(content)) if isinstance(content, dict) else self._preprocess_text(content) for _, content in articles]
            dictionary = corpora.Dictionary(processed_articles)
            corpus = [dictionary.doc2bow(text) for text in processed_articles]
            
            lda = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=len(articles),
                passes=15,
                random_state=42
            )

            # 5. Generate and save HTML (with explicit verification)
            vis_path = output_dir / "analysis.html"
            pyLDAvis.save_html(
                pyLDAvis.gensim_models.prepare(lda, corpus, dictionary),
                str(vis_path)
            )

            print(f"Successfully generated: {vis_path}")
            return str(vis_path)

        except Exception as e:
            print(f"Failed to generate visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

    def _generate_loophole_html(self, draft: Dict) -> str:
        """Generate HTML for loophole findings"""
        analysis = self.loophole_analyzer.analyze_draft(draft)
        cards = []

        for article, data in analysis['by_article'].items():
            if data['findings']:
                cards.append(f"""
                <div class="analysis-card">
                    <h3>{html.escape(article)}</h3>
                    <p><strong>Score:</strong> {data['score']}/100</p>
                    {"".join(
                        f'<p><strong>{cat}:</strong> {html.escape(str(f)[:100])}...</p>'
                        for cat, findings in data['findings'].items()
                        for f in findings
                    )}
                </div>
                """)

        return f"""
        <div style="background:#f8f9fa; padding:20px; border-radius:8px; margin-bottom:30px">
            <h3 style="margin-top:0">Document Summary</h3>
            <div style="display:flex; gap:20px; margin-bottom:20px">
                <div style="flex:1; background:#f8d7da; padding:15px; border-radius:5px">
                    <h4 style="margin-top:0; color:#dc3545">High Risk Issues</h4>
                    <p style="font-size:24px; margin:10px 0; text-align:center">{analysis['stats']['high_risk']}</p>
                </div>
                <div style="flex:1; background:#e2e3e5; padding:15px; border-radius:5px">
                    <h4 style="margin-top:0; color:#6c757d">Total Findings</h4>
                    <p style="font-size:24px; margin:10px 0; text-align:center">{analysis['stats']['total_loopholes']}</p>
                </div>
            </div>

            <h3>Article-Level Findings</h3>
            <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(400px, 1fr)); gap:20px">
                {''.join(cards)}
            </div>

            {self._generate_cross_article_html(analysis['cross_article'])}
        </div>
        """

    def _generate_article_card(self, article: str, data: Dict) -> str:
        """Generate HTML card for article findings"""
        return f"""
        <div style="background:white; border-radius:5px; padding:15px; box-shadow:0 2px 5px rgba(0,0,0,0.1)">
            <h4 style="margin-top:0; color:#2c3e50">{html.escape(article)}</h4>
            <p style="color:#6c757d">Found {data['count']} potential issues</p>
            <div style="max-height:200px; overflow-y:auto; border-top:1px solid #eee; padding-top:10px">
                {''.join(self._generate_finding_item(cat, item) for cat, items in data['details'].items() for item in items)}
            </div>
        </div>
        """

    def _generate_finding_item(self, category: str, finding: Dict) -> str:
        """Generate HTML for individual finding"""
        risk_styles = {
            'undefined_terms': ('#dc3545', 'HIGH'),
            'broad_exceptions': ('#dc3545', 'HIGH'),
            'ambiguous_terms': ('#ffc107', 'MEDIUM'),
            'open_conditions': ('#ffc107', 'MEDIUM'),
            'recursive_references': ('#28a745', 'LOW')
        }
        color, level = risk_styles.get(category, ('#6c757d', 'LOW'))

        return f"""
        <div style="margin-bottom:10px; padding-bottom:10px; border-bottom:1px dashed #eee">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <strong>{html.escape(category.replace('_', ' ').title())}</strong>
                <span style="background:{color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.8em">
                    {level}
                </span>
            </div>
            <p style="margin:5px 0"><code>{html.escape(finding['text'][:80])}</code></p>
            <p style="color:#6c757d; font-size:0.9em">{html.escape(finding['exploit'])}</p>
        </div>
        """

    def _generate_cross_article_html(self, issues: List) -> str:
        """Generate HTML for cross-article issues"""
        if not issues:
            return ""

        return f"""
        <h3 style="margin-top:30px">Cross-Article Conflicts</h3>
        <div style="background:white; border-radius:5px; padding:15px; box-shadow:0 2px 5px rgba(0,0,0,0.1)">
            <ul style="margin:0; padding-left:20px">
                {''.join(f'<li style="margin-bottom:8px"><strong>{html.escape(issue["articles"])}:</strong> {html.escape(", ".join(issue["issues"]))}</li>' for issue in issues)}
            </ul>
        </div>
        """

    def _article_loophole_card(self, article: str, data: Dict) -> str:
        """Generate a compact card for article-level findings"""
        return f"""
        <div style="border:1px solid #ddd; padding:15px; border-radius:5px">
            <h4>{article}</h4>
            <p>Issues found: <strong>{data['count']}</strong></p>
            <div style="max-height:150px; overflow-y:auto">
                {''.join(f'<p style="font-size:0.9em; margin:5px 0">• {cat}: {item["text"][:50]}...</p>' 
                        for cat, items in data['details'].items() for item in items)}
            </div>
        </div>
        """

    def _generate_article_visualizations(self, draft_id: str, remove_stopwords: bool = True, lemmatize: bool = True) -> Dict[str, str]:
        """Generate individual HTML visualizations for each article with flexible preprocessing"""
        paths = {}
        for article, content in self.drafts[draft_id]['articles'].items():
            try:
                if len(content.split()) < 30:
                    continue  # Skip short articles

                safe_name = re.sub(r'\W+', '_', article)[:50]
                vis_path = self.report_dir / draft_id / f"{safe_name}_topics.html"

                processed = self._preprocess_text(content, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
                dictionary = corpora.Dictionary([processed])
                corpus = [dictionary.doc2bow(processed)]

                lda = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=1,
                    passes=10,
                    random_state=42
                )

                # Generate and save visualization
                pyLDAvis.save_html(
                    pyLDAvis.gensim_models.prepare(lda, corpus, dictionary),
                    str(vis_path)
                )
                paths[article] = str(vis_path)

                print(f"Article visualization \033[92mdone\033[0m for '{article[:30]}...'")
                
            except Exception as e:
                print(f"Article visualization \033[91mfailed\033[0m for '{article[:30]}...': {str(e)}")
        return paths

    def _preprocess_text(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> List[str]:
        """Text preprocessing for analysis with flexible options"""
        doc = self.nlp(text.lower())
        tokens = []
        for token in doc:
            if remove_stopwords and token.is_stop:
                continue
            if not token.is_alpha:
                continue
            if lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        return tokens

    def _generate_text_report(self, draft: Dict, analysis: Dict) -> str:
        """Generate report showing article-to-topic mapping"""
        report = [
            "ARTICLE-TO-TOPIC ANALYSIS REPORT",
            "="*50,
            f"Document: {draft['filename']}",
            f"Status: {'VALID' if draft['is_valid'] else 'INVALID'}",
            "",
            "TOPIC MAPPING:",
            "-"*50
        ]

        if 'topics' in analysis:
            for topic in analysis['topics']:
                report.append(
                    f"Topic {topic['topic_id']}: {topic['article']} "
                    f"(score: {topic['article_score']:.2f})"
                )
                report.append(f"   Keywords: {', '.join(topic['keywords'][:5])}")
        else:
            report.append("No topic analysis available")

        report.extend([
            "",
            "STATISTICS:",
            f"- Articles: {analysis['stats']['article_count']}",
            f"- Topics: {len(analysis.get('topics', []))}",
            f"- Readability: {analysis['readability']['flesch']:.1f}",
            "="*50
        ])

        return "\n".join(report)
    
    def _generate_analysis_html(self, draft_id: str) -> str:
        """Generate complete tabbed HTML report (formerly _generate_tabbed_report)"""
        draft = self.drafts[draft_id]

        # Generate all analysis components
        lda_html = self._generate_lda_visualization(draft_id)
        comments_html = self._generate_content_comments(draft)
        loopholes_html = self._generate_loophole_html(draft)
        # Add raw input content tab
        import html as html_module  # Import html module with alias to avoid local variable conflict
        raw_content_html = f"<pre style='white-space: pre-wrap; background:#f4f4f4; padding:15px; border-radius:5px; max-height:600px; overflow-y:auto;'>{html_module.escape(draft.get('content', ''))}</pre>"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Legal Analysis: {draft['filename']}</title>
            <style>
                /* Tab Styling */
                .tab-header {{
                    display: flex;
                    border-bottom: 1px solid #ddd;
                    margin-bottom: 20px;
                }}
                .tab-btn {{
                    padding: 10px 20px;
                    cursor: pointer;
                    background: #f1f1f1;
                    border: none;
                    margin-right: 5px;
                    border-radius: 5px 5px 0 0;
                }}
                .tab-btn.active {{
                    background: white;
                    border: 1px solid #ddd;
                    border-bottom: 1px solid white;
                    margin-bottom: -1px;
                    font-weight: bold;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-top: none;
                }}
                .tab-content.active {{
                    display: block;
                }}

                /* Content Styling */
                .analysis-card {{
                    background: white;
                    border-radius: 4px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <h1>Legal Analysis Report</h1>
            <h2>{draft['filename']}</h2>

            <!-- Tab Interface -->
            <div class="tab-header">
                <button class="tab-btn" onclick="showTab('raw_content')">Raw Content</button>
                <button class="tab-btn active" onclick="showTab('topics')">Topics</button>
                <button class="tab-btn" onclick="showTab('comments')">Comments</button>
                <button class="tab-btn" onclick="showTab('loopholes')">Loopholes</button>
            </div>

            <!-- Tab Content -->
            <div id="raw_content" class="tab-content">
                {raw_content_html}
            </div>

            <div id="topics" class="tab-content active">
                {lda_html}
            </div>

            <div id="comments" class="tab-content">
                {comments_html}
            </div>

            <div id="loopholes" class="tab-content">
                {loopholes_html}
            </div>

            <script>
                function showTab(tabId) {{
                    // Hide all tabs
                    document.querySelectorAll('.tab-content').forEach(tab => {{
                        tab.classList.remove('active');
                    }});

                    // Deactivate all buttons
                    document.querySelectorAll('.tab-btn').forEach(btn => {{
                        btn.classList.remove('active');
                    }});

                    // Activate selected tab
                    document.getElementById(tabId).classList.add('active');
                    event.currentTarget.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """

        # Save the report
        report_path = self.report_dir / draft_id / "analysis.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(report_path)

    def _generate_content_comments(self, draft: Dict) -> str:
        """Generate extended and clearer HTML for content commentary focused on potential loopholes"""
        comments = []
        # Flatten article content if nested dicts to avoid errors in loophole analysis
        draft_flattened = draft.copy()
        draft_flattened['articles'] = {k: self._flatten_article_content(v) for k, v in draft.get('articles', {}).items()}
        analysis = self.loophole_analyzer.analyze_draft(draft_flattened)
        for article, data in analysis['by_article'].items():
            # Skip grouping titles in comments as well
            if self._is_grouping_title(article):
                continue
            # Skip articles with no findings or empty findings dict
            if not data.get('findings') or all(not v for v in data['findings'].values()):
                continue
            comment_items = []
            for category, findings in data['findings'].items():
                if not findings:
                    continue
                for f in findings:
                    # Normalize finding if it's a dict or string
                    if isinstance(f, dict):
                        text = f.get("text", "")
                        context = f.get("context", "")
                        issue = f.get("issue", "")
                        severity = f.get("severity", "")
                    else:
                        text = str(f)
                        context = ""
                        issue = ""
                        severity = ""
                    # Add exploit description if missing
                    exploit = f.get("exploit", "") if isinstance(f, dict) else ""
                    comment_text = f"<strong>Issue:</strong> {html.escape(issue)}<br>" if issue else ""
                    comment_text += f"<strong>Severity:</strong> {html.escape(severity)}<br>" if severity else ""
                    #comment_text += f"<strong>Text:</strong> {html.escape(text)}<br>"
                    if exploit:
                        comment_text += f"<strong>Exploit:</strong> {html.escape(exploit)}<br>"
                    if context:
                        comment_text += f"<strong>Context:</strong> {html.escape(context)}<br>"
                    comment_items.append(f"<li>{comment_text}</li>")
            comments.append(f"""
            <div class="analysis-card">
                <h3>{html.escape(article)}</h3>
                <ul>
                    {"".join(comment_items)}
                </ul>
            </div>
            """)
        return "\n".join(comments) if comments else "<p>No potential loopholes detected</p>"

    def _generate_loophole_html(self, draft: Dict) -> str:
        """Generate HTML for loophole analysis"""
        analysis = self.loophole_analyzer.analyze_draft(draft)
        cards = []

        for article, data in analysis['by_article'].items():
            if data['findings']:
                cards.append(f"""
                <div class="analysis-card">
                    <h3>{html.escape(article)}</h3>
                    <p><strong>Score:</strong> {data['score']}/100</p>
                    {"".join(
                        f'<p><strong>{cat}:</strong> {html.escape(str(f)[:100])}...</p>'
                        for cat, findings in data['findings'].items()
                        for f in findings
                    )}
                </div>
                """)

        return "\n".join(cards) if cards else "<p>No significant loopholes detected</p>"

    def _analyze_article_content(self, text: str) -> Dict:
        """Analyze article text for substantive issues"""
        comments = []

        # Example analysis - expand with your specific legal checks
        if len(re.findall(r'\bhowever\b|\bnotwithstanding\b', text, re.IGNORECASE)) > 3:
            comments.append("Excessive use of qualifying language may create ambiguity")

        if text.count(';') > text.count('.'):
            comments.append("High semicolon usage suggests complex, potentially unclear sentence structures")

        if 'may' in text.lower() and 'shall' in text.lower():
            comments.append("Mixed use of permissive ('may') and mandatory ('shall') language")

        return {
            'comments': comments,
            'word_count': len(text.split())
        }
    
    def _verify_file_creation(self, filepath: str) -> bool:
        """Explicit file creation checker"""
        path = Path(filepath)
        if not path.exists():
            print(f"File not created: {filepath}")
            return False

        if path.stat().st_size == 0:
            print(f"Empty file created: {filepath}")
            return False

        print(f"Verified valid file: {filepath} ({path.stat().st_size} bytes)")
        return True
    
    def _sanitize_lda_data(self, data):
        """Recursively convert complex numpy types to floats in LDA data for JSON serialization"""
        import numpy as np
    
        if isinstance(data, np.complexfloating):
            return float(data.real)
        elif isinstance(data, dict):
            return {k: self._sanitize_lda_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_lda_data(item) for item in data]
        # Leave tuples unchanged to preserve pyLDAvis structure
        else:
            return data
    
    
    def _generate_lda_visualization(self, draft_id: str, num_topics: int = None, passes: int = 15, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        """Generate LDA visualization HTML with flexible parameters"""
        try:
            draft = self.drafts[draft_id]
            articles = [(title, content) for title, content in draft['articles'].items() if not self._is_grouping_title(title)]
    
            print(f"DEBUG _generate_lda_visualization: Number of articles: {len(articles)}")

            if not articles:
                return "<p>No articles available for visualization</p>"

            processed_articles = [self._preprocess_text(self._flatten_article_content(content), remove_stopwords=remove_stopwords, lemmatize=lemmatize) for _, content in articles]
            dictionary = corpora.Dictionary(processed_articles)
            corpus = [dictionary.doc2bow(text) for text in processed_articles]

            topic_count = num_topics if num_topics is not None else len(articles)
            print(f"DEBUG _generate_lda_visualization: Number of topics: {topic_count}")

            # If only one topic, add a dummy document to avoid pyLDAvis MDS assertion error
            if topic_count == 1:
                processed_articles.append([])  # Add empty document as dummy
                dictionary = corpora.Dictionary(processed_articles)
                corpus = [dictionary.doc2bow(text) for text in processed_articles]
                lda = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=2,
                    passes=passes
                )
            else:
                dictionary = corpora.Dictionary(processed_articles)
                corpus = [dictionary.doc2bow(text) for text in processed_articles]
                lda = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=topic_count,
                    passes=passes
                )
    
            vis_data = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
            sanitized_vis_data = self._sanitize_lda_data(vis_data)
            return pyLDAvis.prepared_data_to_html(sanitized_vis_data)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Visualization generation failed with error: {str(e)}\nTraceback:\n{tb}")
            # Return detailed error and traceback in HTML output
            error_html = f"<p>Visualization failed: {html.escape(str(e))}</p><pre style='white-space: pre-wrap; background:#f8d7da; padding:10px; border-radius:5px; overflow-x:auto;'>{html.escape(tb)}</pre>"
            return error_html

    def analyze_overgeneralization(self, draft_id: str) -> Dict[str, float]:
        """
        Analyze overgeneralization in topic modeling by computing average topic coherence
        and average keyword overlap between topics.
        Returns a dictionary with 'average_coherence' and 'average_keyword_overlap'.
        """
        draft = self.drafts.get(draft_id)
        if not draft:
            raise ValueError(f"Draft ID {draft_id} not found")

        articles = list(draft['articles'].items())
        if not articles:
            return {'average_coherence': 0.0, 'average_keyword_overlap': 0.0}

        processed_articles = [self._preprocess_text(content) for _, content in articles]
        dictionary = corpora.Dictionary(processed_articles)
        corpus = [dictionary.doc2bow(text) for text in processed_articles]

        num_topics = len(articles)
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=15,
            random_state=42,
            alpha='auto'
        )

        # Compute topic coherence scores
        coherences = []
        for bow in corpus:
            doc_topics = lda.get_document_topics(bow)
            max_coherence = max([score for _, score in doc_topics], default=0.0)
            coherences.append(max_coherence)
        average_coherence = float(np.mean(coherences)) if coherences else 0.0

        # Compute average keyword overlap between topics
        topic_keywords = []
        for topic_id in range(num_topics):
            keywords = set(word for word, _ in lda.show_topic(topic_id))
            topic_keywords.append(keywords)

        total_overlap = 0
        count = 0
        for i in range(num_topics):
            for j in range(i + 1, num_topics):
                overlap = len(topic_keywords[i].intersection(topic_keywords[j]))
                total_overlap += overlap
                count += 1
        average_keyword_overlap = total_overlap / count if count > 0 else 0.0

        return {
            'average_coherence': average_coherence,
            'average_keyword_overlap': average_keyword_overlap
        }
    


if __name__ == "__main__":
    system = LawDraftSystem()
    
    try:
        files = system.importer.get_available_files()
        if files:
            print(f"Found {len(files)} documents to analyze")
            if len(files) == 1:
                selected_file = files[0]['path']
            else:
                print("Multiple input files found. Please select one to analyze:")
                for idx, file in enumerate(files):
                    print(f"{idx + 1}: {file['path']}")
                while True:
                    choice = input(f"Enter a number (1-{len(files)}): ")
                    if choice.isdigit():
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(files):
                            selected_file = files[choice_num - 1]['path']
                            break
                    print("Invalid choice. Please try again.")
            draft_id = system.import_draft(selected_file)
            
            # Add this verification step
            report_dir = system.report_dir / draft_id
            print("\n=== Generated Files ===")
            for f in report_dir.glob('*'):
                print(f"- {f.name} ({f.stat().st_size} bytes)")
            
            # Attempt to open in browser
            try:
                import webbrowser
                report_path = report_dir / "analysis.html"
                if report_path.exists():
                    webbrowser.open(f"file://{report_path.resolve()}")
            except Exception as e:
                print(f"Browser open failed: {str(e)}")
        else:
            print("No files found in input_drafts directory")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
