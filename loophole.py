import re
import spacy
from typing import Dict, List, Tuple
from collections import defaultdict
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LoopholeAnalyzer:
    """Enhanced legal draft analyzer with substantive commentary and dynamic detection"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.TextBlob = TextBlob
        self.defaultdict = defaultdict
        self.patterns = {
            'vagueness': [
                r'\b(reasonable|appropriate|substantial)\b(?!\s+standard)',
                r'\b(may\s+at\s+its\s+discretion)\b',
                r'\b(as\s+determined\s+by)\b'
            ],
            'contradictions': [
                r'(notwithstanding[^\.,;]+(shall|must))',
                r'(except\s+as\s+otherwise\s+provided)[^\.,;]+(shall)',
                r'(where\s+applicable)[^\.,;]+(must\s+not)'
            ],
            'rights_issues': [
                r'(right(s)?\s+to\s+[^\.,;]+(?!\s+(shall|must)))',
                r'(no\s+right(s)?\s+to\s+[^\.,;]+(?!\s+create))'
            ],
            'undefined_terms': [
                
            ],
            'broad_exceptions': [
                r'except as provided by law',
                r'except as otherwise provided'
            ],
            'ambiguous_terms': [
                r'\breasonable\b',
                r'\bappropriate\b',
                r'\bsubstantial\b'
            ],
            'open_conditions': [
                r'subject to',
                r'unless otherwise'
            ],
            'recursive_references': [
                r'as defined in this section',
                r'as set forth herein'
            ]
        }

    def _flatten_article_content(self, content) -> str:
        """Flatten nested article content dict to a single string for analysis"""
        if isinstance(content, dict):
            parts = []
            for key, value in content.items():
                if isinstance(value, dict):
                    parts.append(self._flatten_article_content(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            parts.append(self._flatten_article_content(item))
                        else:
                            parts.append(str(item))
                else:
                    parts.append(str(value))
            return ' '.join(parts)
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(self._flatten_article_content(item))
                else:
                    parts.append(str(item))
            return ' '.join(parts)
        else:
            return str(content)

    @staticmethod
    def _is_grouping_title(title: str) -> bool:
        """Detect if a title is a grouping like 'part 1', 'chapter 1', 'section 1' etc."""
        pattern = re.compile(r'^\s*(part|chapter|section|title|subtitle|division|book)\s*\d+', re.IGNORECASE)
        return bool(pattern.match(title.strip()))

    def analyze_draft(self, draft: Dict) -> Dict[str, Dict]:
        """Comprehensive analysis with article-level and cross-document insights"""
        self._processed_articles = {title: content for title, content in draft.get('articles', {}).items() if not self._is_grouping_title(title)}

        results = {
            'by_article': {},
            'cross_analysis': {
                'contradictions': [],
                'definition_consistency': self._check_definitions(draft)
            },
            'metadata': {
                'total_articles': len(self._processed_articles),
                'avg_loophole_score': 0
            }
        }
        total_scores = 0
        
        for title, content in self._processed_articles.items():
            logger.debug(f"Analyzing article: {title}")
            article_results = self.analyze_article(content)
            results['by_article'][title] = article_results
            total_scores += article_results['score']
            
            # Cross-article checks
            self._find_cross_article_issues(title, content, results['cross_analysis'])

        if results['metadata']['total_articles'] > 0:
            results['metadata']['avg_loophole_score'] = round(
                total_scores / results['metadata']['total_articles'], 2)

        logger.debug(f"Draft analysis complete: {results}")
        return results

    def analyze_draft(self, draft: Dict) -> Dict[str, Dict]:
        """Comprehensive analysis with article-level and cross-document insights"""
        self._processed_articles = draft.get('articles', {})

        results = {
            'by_article': {},
            'cross_analysis': {
                'contradictions': [],
                'definition_consistency': self._check_definitions(draft)
            },
            'metadata': {
                'total_articles': len(self._processed_articles),
                'avg_loophole_score': 0
            }
        }
        total_scores = 0
        
        for title, content in draft.get('articles', {}).items():
            logger.debug(f"Analyzing article: {title}")
            if isinstance(content, dict):
                content_str = self._flatten_article_content(content)
            else:
                content_str = content
            article_results = self.analyze_article(content_str)
            results['by_article'][title] = article_results
            total_scores += article_results['score']
            
            # Cross-article checks
            self._find_cross_article_issues(title, content, results['cross_analysis'])

        if results['metadata']['total_articles'] > 0:
            results['metadata']['avg_loophole_score'] = round(
                total_scores / results['metadata']['total_articles'], 2)

        logger.debug(f"Draft analysis complete: {results}")
        return results

    def analyze_article(self, text: str) -> Dict:
        """Multidimensional analysis of a single article"""
        logger.debug(f"Analyzing article text of length {len(text)}")
        doc = self.nlp(text)
        raw_findings = self.defaultdict(list)
        suggestions = []
        
        # Pattern-based detection
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    context = self._get_context(text, match.start(), match.end())
                    # Compose issue description based on category and matched text
                    issue_desc = f"{category.replace('_', ' ').title()}: '{match.group()}'"
                    # Avoid redundant flags for same issue and context
                    if any(f['issue'] == issue_desc and f['context'] == context for f in raw_findings[category]):
                        continue
                    raw_findings[category].append({
                        'severity': self._assess_severity(category, context),
                        'context': context,
                        'issue': issue_desc
                    })
                    logger.debug(f"Found {category} issue: {issue_desc} at position {match.start()}-{match.end()}")

        # Aggregate repeated findings by issue and context to reduce noise
        findings = self.defaultdict(list)
        for category, items in raw_findings.items():
            seen_issues = set()
            for item in items:
                key = (item['issue'].lower(), item['context'])
                if key not in seen_issues:
                    count = sum(1 for i in items if i['issue'].lower() == item['issue'].lower() and i['context'] == item['context'])
                    item_copy = item.copy()
                    if count > 1:
                        item_copy['count'] = count
                        # Adjust issue description to avoid redundancy
                        item_copy['issue'] = item_copy['issue'].replace(' (occurred', f' (occurred {count} times')
                    # Use a set to avoid duplicate entries in findings[category]
                    if item_copy not in findings[category]:
                        findings[category].append(item_copy)
                    seen_issues.add(key)

        # Structural analysis
        # Detect overly long sentences with multiple subprovisions separated by semicolons or numbering
        sentences = list(doc.sents)
        for sent in sentences:
            # Heuristic: if sentence contains multiple semicolons or numbered subclauses, flag as complex
            semicolon_count = sent.text.count(';')
            numbered_subclauses = len(re.findall(r'\b\d+[\.\)]', sent.text))
            if semicolon_count > 2 or numbered_subclauses > 2:
               # Attempt to split sentence into subprovisions for clearer reporting
               # Split by semicolons and numbered subclauses, preserving numbering
                subprovisions = re.split(r';\s*|\s*(?=\d+\.)', sent.text)
                # Check if subprovisions are effectively divided
                if len(subprovisions) > 1:
                    for subprov in subprovisions:
                        subprov = subprov.strip()
                        if len(subprov) > 50:  # Only report substantial subprovisions
                            findings['structure'].append({
                                'severity': 'medium',
                                'context': subprov,
                                'issue': 'Subprovision detected within complex sentence; consider dividing for clarity'
                            })
                            logger.debug(f"Detected subprovision: {subprov}")
                else:
                    # If not divided, append the whole sentence as complex
                    findings['structure'].append({
                        'severity': 'medium',
                        'context': sent.text.strip(),
                        'issue': 'Complex sentence with multiple subprovisions detected; consider dividing for clarity',
                        'suppress_if_divided': True
                    })
                    logger.debug(f"Detected complex sentence with subprovisions: {sent.text.strip()}")

        # Detect logical structures like if-then, and, or, not, then

        logical_issues = []
        # Analyze relationships and dependencies between clauses connected by logical operators
        # Flag potential ambiguities or inconsistencies in how logical connectors combine phrases
        # Provide suggestions or warnings only when the logical structure is unclear or potentially problematic

        # Check if text contains subsets of articles (e.g., numbered list or multiple paragraphs)
        # If so, skip adding the complex logical structure issue
        has_subsets = False
        # Heuristic: check for multiple numbered points or paragraphs
        if re.search(r'\n\s*\d+[\.\)]', text) or re.search(r'\n\s*[a-zA-Z][\.\)]', text):
            has_subsets = True

        sentences = list(self.nlp(text).sents)
        for i, sent in enumerate(sentences):
            sent_text = sent.text.lower()
            connectors_in_sent = [conn for conn in ['if', 'and', 'or', 'not', 'then'] if f' {conn} ' in sent_text]
            if connectors_in_sent:
                # Remove 'or' from being flagged as an issue, treat as neutral connector
                connectors_filtered = [conn for conn in connectors_in_sent if conn != 'or']
                if not connectors_filtered:
                    # Only 'or' present, do not flag as issue
                    continue
                # Check for ambiguous or inconsistent use of connectors
                ambiguous = False
                if 'if' in connectors_filtered and ('and' in connectors_filtered):
                    ambiguous = True
                if ambiguous:
                    logical_issues.append({
                        'severity': 'medium',
                        'context': sent.text.strip(),
                        'issue': f"Ambiguous logical connectors detected: {', '.join(connectors_filtered)}"
                    })
                    logger.debug(f"Ambiguous logical connectors in sentence: {sent.text.strip()}")
                else:
                    logical_issues.append({
                        'severity': 'low',
                        'context': sent.text.strip(),
                        'issue': f"Logical connectors detected: {', '.join(connectors_filtered)}"
                    })
                    logger.debug(f"Logical connectors in sentence: {sent.text.strip()}")

        # Analyze logical structure complexity
        if len(logical_issues) > 5 and not has_subsets:
            logical_issues.append({
                'severity': 'medium',
                'context': text,
                'issue': 'Complex logical structure detected with multiple connectors'
            })
            logger.debug("Detected complex logical structure with multiple connectors")

        # Detect compound logical structures and incomplete logic (e.g., "not answered" states)
        compound_logic_patterns = [
            r'IF\s*\(.*AND.*\).*THEN',
            r'IF\s*\(.*OR.*\).*THEN',
            r'ELSE\s*IF\s*\(.*\)',
            r'NOT\s*ANSWERED',
            r'UNANSWERED'
        ]
        for pattern in compound_logic_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                context = self._get_context(text, match.start(), match.end())
                logical_issues.append({
                    'severity': 'high',
                    'context': context,
                    'issue': f"Compound or incomplete logical structure detected: '{match.group()}'"
                })
                logger.debug(f"Detected compound logical structure: {match.group()} at position {match.start()}-{match.end()}")

        findings['logical_structure'].extend(logical_issues)

        # Sentiment analysis for obligations
        obligation_score = self._analyze_obligation_sentiment(text)
        logger.debug(f"Obligation sentiment score: {obligation_score}")
        if obligation_score < -0.3:
            findings['rights_balance'].append({
                'severity': 'high',
                'context': text,
                'issue': 'Overly negative obligation language'
            })
            logger.debug("Detected overly negative obligation language")

        # Generate suggestions
        # Remove complex sentence issues if subprovisions are detected
        if 'structure' in findings:
            subprovision_issues = [f for f in findings['structure'] if 'Subprovision detected' in f['issue']]
            complex_issues = [f for f in findings['structure'] if 'Complex sentence with multiple subprovisions' in f['issue']]
            if subprovision_issues and complex_issues:
                # Remove complex issues that have suppress_if_divided flag
                findings['structure'] = [f for f in findings['structure'] if not (f in complex_issues and f.get('suppress_if_divided'))]
            else:
                # If subprovisions are not detected, keep complex issues
                findings['structure'] = [f for f in findings['structure'] if f not in complex_issues or not f.get('suppress_if_divided')]

        if findings.get('vagueness'):
            suggestions.append("Replace vague terms with quantifiable standards")
        if findings.get('contradictions'):
            suggestions.append("Clarify apparent contradictions in requirements")
        if findings.get('logical_structure'):
            suggestions.append("Review logical connectors for clarity and correctness")

        # Calculate composite score (100 = perfect)
        severity_weights = {'high': 3, 'medium': 2, 'low': 1}
        score = max(0, 100 - sum(
            severity_weights[f['severity'] ]
            for category in findings.values() 
            for f in category
        ) * 5)

        logger.debug(f"Article score: {score}, suggestions: {suggestions}")

        return {
            'findings': dict(findings),
            'score': score,
            'suggestions': suggestions or ["No critical issues detected"],
            'word_count': len(text.split())
        }

    def _find_cross_article_issues(self, current_title: str, current_content: str, results: Dict):
        """Identify conflicts between articles"""
        if isinstance(current_content, dict):
            current_content_str = self._flatten_article_content(current_content)
        else:
            current_content_str = current_content
        current_entities = {ent.text.lower() for ent in self.nlp(current_content_str).ents 
                      if ent.label_ in ['LAW', 'ORG']}
    
        # Use the stored processed articles
        for other_title, other_content in self._processed_articles.items():
            if other_title == current_title:
                continue
            if isinstance(other_content, dict):
                other_content_str = self._flatten_article_content(other_content)
            else:
                other_content_str = other_content
            other_entities = {ent.text.lower() for ent in self.nlp(other_content_str).ents 
                            if ent.label_ in ['LAW', 'ORG']}
            
            common_entities = current_entities & other_entities
            for entity in common_entities:
                if ('shall not' in current_content_str and 'shall' in other_content_str) or \
                   ('prohibited' in current_content_str and 'permitted' in other_content_str):
                    results['contradictions'].append({
                        'articles': f"{current_title} vs {other_title}",
                        'entity': entity,
                        'conflict': "Contradictory requirements"
                    })
                    logger.debug(f"Cross-article contradiction found between {current_title} and {other_title} on entity {entity}")

    def _check_definitions(self, draft: Dict) -> List:
        """Verify consistent use of defined terms"""
        definition_terms = set()
        for title, content in draft.get('articles', {}).items():
            # Flatten content if it is a dict to string
            if isinstance(content, dict):
                content_str = self._flatten_article_content(content)
            else:
                content_str = content
            if "means" in content_str.lower() or "defined as" in content_str.lower():
                definition_terms.update(re.findall(r'"(.*?)"', content_str))
        
        inconsistencies = []
        for term in definition_terms:
            uses = 0
            definitions = 0
            for content in draft.get('articles', {}).values():
                if isinstance(content, dict):
                    content_str = self._flatten_article_content(content)
                else:
                    content_str = content
                uses += content_str.count(term)
                if f'"{term}"' in content_str and ("means" in content_str.lower() or "defined as" in content_str.lower()):
                    definitions += 1
            
            if uses > 0 and definitions == 0:
                inconsistencies.append(f'Undefined term: "{term}"')

        logger.debug(f"Definition inconsistencies found: {inconsistencies}")
        return inconsistencies

    def _analyze_obligation_sentiment(self, text: str) -> float:
        """Assess whether obligations are phrased positively/negatively"""
        obligations = [sent.text for sent in self.nlp(text).sents 
                      if 'shall' in sent.text.lower() or 'must' in sent.text.lower()]
        if not obligations:
            return 0
        polarity_sum = 0
        for oblig in obligations:
            polarity = self.TextBlob(oblig).sentiment.polarity
            logger.debug(f"Obligation sentiment polarity: {polarity} for sentence: {oblig}")
            polarity_sum += polarity
        return polarity_sum / len(obligations)

    def _assess_severity(self, category: str, context: str) -> str:
        """Determine severity based on context"""
        if category == 'contradictions':
            return 'high'
        if 'right' in context.lower() and 'obligation' not in context.lower():
            return 'high'
        return 'medium' if category == 'vagueness' else 'low'

    def _get_context(self, text: str, start: int, end: int, window=100) -> str:
        """Extract the full sentence containing the matched text for better context"""
        doc = self.nlp(text)
        for sent in doc.sents:
            if sent.start_char <= start and sent.end_char >= end:
                return sent.text.strip()
        # Fallback to original method if sentence not found
        return text[max(0, start-window):min(len(text), end+window)]
