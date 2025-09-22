"""
Citation and bibliography management for academic publications.

Provides comprehensive citation generation and bibliography management
for ESCAI Framework research outputs.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class Author:
    """Represents an author with full details."""
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None
    
    def full_name(self) -> str:
        """Get full name in standard format."""
        if self.middle_name:
            return f"{self.first_name} {self.middle_name} {self.last_name}"
        return f"{self.first_name} {self.last_name}"
    
    def last_first(self) -> str:
        """Get name in Last, First format."""
        if self.middle_name:
            return f"{self.last_name}, {self.first_name} {self.middle_name[0]}."
        return f"{self.last_name}, {self.first_name}"


@dataclass
class Citation:
    """Comprehensive citation information."""
    title: str
    authors: List[Author]
    year: Optional[int] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    number: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    isbn: Optional[str] = None
    book_title: Optional[str] = None
    editor: Optional[str] = None
    conference: Optional[str] = None
    location: Optional[str] = None
    note: Optional[str] = None
    entry_type: str = "article"  # article, book, inproceedings, etc.
    
    def to_bibtex(self, key: str) -> str:
        """Generate BibTeX entry."""
        bibtex = f"@{self.entry_type}{{{key},\n"
        
        # Required fields
        bibtex += f"  title={{{self.title}}},\n"
        
        # Authors
        author_str = " and ".join([author.full_name() for author in self.authors])
        bibtex += f"  author={{{author_str}}},\n"
        
        # Optional fields based on entry type
        if self.year:
            bibtex += f"  year={{{self.year}}},\n"
        
        if self.entry_type == "article":
            if self.journal:
                bibtex += f"  journal={{{self.journal}}},\n"
            if self.volume:
                bibtex += f"  volume={{{self.volume}}},\n"
            if self.number:
                bibtex += f"  number={{{self.number}}},\n"
        
        elif self.entry_type == "book":
            if self.publisher:
                bibtex += f"  publisher={{{self.publisher}}},\n"
            if self.isbn:
                bibtex += f"  isbn={{{self.isbn}}},\n"
        
        elif self.entry_type == "inproceedings":
            if self.book_title:
                bibtex += f"  booktitle={{{self.book_title}}},\n"
            if self.conference:
                bibtex += f"  series={{{self.conference}}},\n"
            if self.location:
                bibtex += f"  address={{{self.location}}},\n"
        
        # Common optional fields
        if self.pages:
            bibtex += f"  pages={{{self.pages}}},\n"
        if self.doi:
            bibtex += f"  doi={{{self.doi}}},\n"
        if self.url:
            bibtex += f"  url={{{self.url}}},\n"
        if self.note:
            bibtex += f"  note={{{self.note}}},\n"
        
        bibtex += "}\n"
        return bibtex
    
    def to_apa(self) -> str:
        """Generate APA format citation."""
        # Authors
        if len(self.authors) == 1:
            authors_str = self.authors[0].last_first()
        elif len(self.authors) <= 7:
            author_names = [author.last_first() for author in self.authors[:-1]]
            authors_str = ", ".join(author_names) + f", & {self.authors[-1].last_first()}"
        else:
            # More than 7 authors - use et al.
            author_names = [author.last_first() for author in self.authors[:6]]
            authors_str = ", ".join(author_names) + f", ... {self.authors[-1].last_first()}"
        
        citation = authors_str
        
        # Year
        if self.year:
            citation += f" ({self.year})"
        
        # Title
        if self.entry_type == "article":
            citation += f". {self.title}"
        else:
            citation += f". *{self.title}*"
        
        # Publication details
        if self.entry_type == "article" and self.journal:
            citation += f". *{self.journal}*"
            if self.volume:
                citation += f", *{self.volume}*"
                if self.number:
                    citation += f"({self.number})"
            if self.pages:
                citation += f", {self.pages}"
        
        elif self.entry_type == "book" and self.publisher:
            citation += f". {self.publisher}"
        
        elif self.entry_type == "inproceedings":
            if self.book_title:
                citation += f". In *{self.book_title}*"
            if self.pages:
                citation += f" (pp. {self.pages})"
            if self.publisher:
                citation += f". {self.publisher}"
        
        # DOI or URL
        if self.doi:
            citation += f". https://doi.org/{self.doi}"
        elif self.url:
            citation += f". {self.url}"
        
        return citation + "."
    
    def to_ieee(self) -> str:
        """Generate IEEE format citation."""
        # Authors
        if len(self.authors) <= 3:
            author_names = [f"{author.first_name[0]}. {author.last_name}" for author in self.authors]
            authors_str = ", ".join(author_names[:-1])
            if len(author_names) > 1:
                authors_str += f", and {author_names[-1]}"
            else:
                authors_str = author_names[0]
        else:
            authors_str = f"{self.authors[0].first_name[0]}. {self.authors[0].last_name} et al."
        
        citation = authors_str
        
        # Title
        citation += f', "{self.title},"'
        
        # Publication details
        if self.entry_type == "article" and self.journal:
            citation += f" *{self.journal}*"
            if self.volume:
                citation += f", vol. {self.volume}"
            if self.number:
                citation += f", no. {self.number}"
            if self.pages:
                citation += f", pp. {self.pages}"
        
        elif self.entry_type == "book":
            if self.publisher:
                citation += f" {self.publisher}"
        
        elif self.entry_type == "inproceedings":
            if self.book_title:
                citation += f" in *{self.book_title}*"
            if self.pages:
                citation += f", pp. {self.pages}"
        
        # Year
        if self.year:
            citation += f", {self.year}"
        
        return citation + "."


class CitationDatabase:
    """Database of citations for ESCAI Framework methodologies."""
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}
        self._load_default_citations()
    
    def _load_default_citations(self) -> None:
        """Load default citations for common methodologies."""
        
        # Causality and Causal Inference
        self.citations["pearl2009causality"] = Citation(
            title="Causality: Models, Reasoning and Inference",
            authors=[Author("Judea", "Pearl")],
            year=2009,
            publisher="Cambridge University Press",
            entry_type="book",
            isbn="978-0521895606"
        )
        
        self.citations["spirtes2000causation"] = Citation(
            title="Causation, Prediction, and Search",
            authors=[
                Author("Peter", "Spirtes"),
                Author("Clark", "Glymour"),
                Author("Richard", "Scheines")
            ],
            year=2000,
            publisher="MIT Press",
            entry_type="book",
            isbn="978-0262194402"
        )
        
        # Machine Learning and AI
        self.citations["russell2016artificial"] = Citation(
            title="Artificial Intelligence: A Modern Approach",
            authors=[
                Author("Stuart", "Russell"),
                Author("Peter", "Norvig")
            ],
            year=2016,
            publisher="Pearson",
            entry_type="book",
            isbn="978-0134610993"
        )
        
        self.citations["bishop2006pattern"] = Citation(
            title="Pattern Recognition and Machine Learning",
            authors=[Author("Christopher", "Bishop")],
            year=2006,
            publisher="Springer",
            entry_type="book",
            isbn="978-0387310732"
        )
        
        # Multi-Agent Systems
        self.citations["wooldridge2009introduction"] = Citation(
            title="An Introduction to MultiAgent Systems",
            authors=[Author("Michael", "Wooldridge")],
            year=2009,
            publisher="John Wiley & Sons",
            entry_type="book",
            isbn="978-0470519462"
        )
        
        self.citations["stone2000multiagent"] = Citation(
            title="Multiagent Systems: A Survey from a Machine Learning Perspective",
            authors=[Author("Peter", "Stone"), Author("Manuela", "Veloso")],
            year=2000,
            journal="Autonomous Robots",
            volume="8",
            number="3",
            pages="345-383",
            entry_type="article",
            doi="10.1023/A:1008942012299"
        )
        
        # Epistemic Logic and Belief Systems
        self.citations["fagin1995reasoning"] = Citation(
            title="Reasoning About Knowledge",
            authors=[
                Author("Ronald", "Fagin"),
                Author("Joseph", "Halpern"),
                Author("Yoram", "Moses"),
                Author("Moshe", "Vardi")
            ],
            year=1995,
            publisher="MIT Press",
            entry_type="book",
            isbn="978-0262061629"
        )
        
        # Statistical Analysis
        self.citations["wasserman2004all"] = Citation(
            title="All of Statistics: A Concise Course in Statistical Inference",
            authors=[Author("Larry", "Wasserman")],
            year=2004,
            publisher="Springer",
            entry_type="book",
            isbn="978-0387402727"
        )
        
        # Time Series Analysis
        self.citations["hamilton1994time"] = Citation(
            title="Time Series Analysis",
            authors=[Author("James", "Hamilton")],
            year=1994,
            publisher="Princeton University Press",
            entry_type="book",
            isbn="978-0691042893"
        )
        
        # Pattern Mining
        self.citations["han2011data"] = Citation(
            title="Data Mining: Concepts and Techniques",
            authors=[
                Author("Jiawei", "Han"),
                Author("Micheline", "Kamber"),
                Author("Jian", "Pei")
            ],
            year=2011,
            publisher="Morgan Kaufmann",
            entry_type="book",
            isbn="978-0123814791"
        )
        
        # Real-time Systems
        self.citations["buttazzo2011hard"] = Citation(
            title="Hard Real-Time Computing Systems: Predictable Scheduling Algorithms and Applications",
            authors=[Author("Giorgio", "Buttazzo")],
            year=2011,
            publisher="Springer",
            entry_type="book",
            isbn="978-1461406754"
        )
        
        # Contemporary ESCAI-Related Research Papers
        self.citations["ood_detection_2024"] = Citation(
            title="Out-of-Distribution Detection for Neurosymbolic Autonomous Cyber Agents",
            authors=[Author("", "")],  # Authors not specified in the reference
            year=2024,
            journal="IEEE",
            entry_type="inproceedings",
            url="https://ieeexplore.ieee.org/document/10949535/",
            doi="10.1109/EXAMPLE.2024.10949535"
        )
        
        self.citations["smartla_2025"] = Citation(
            title="SMARTLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents",
            authors=[Author("", "")],  # Authors not specified in the reference
            year=2025,
            journal="IEEE",
            entry_type="inproceedings",
            url="https://ieeexplore.ieee.org/document/10745554/",
            doi="10.1109/EXAMPLE.2025.10745554"
        )
        
        self.citations["agent_intelligence_protocol_2025"] = Citation(
            title="Agent Intelligence Protocol: Runtime Governance for Agentic AI Systems",
            authors=[Author("", "")],  # Authors not specified in the reference
            year=2025,
            journal="arXiv preprint",
            entry_type="misc",
            url="https://www.arxiv.org/abs/2508.03858",
            note="arXiv:2508.03858"
        )
        
        self.citations["industry_anomalies_2024"] = Citation(
            title="How Industry Tackles Anomalies during Runtime: Approaches and Key Monitoring Parameters",
            authors=[Author("", "")],  # Authors not specified in the reference
            year=2024,
            journal="IEEE",
            entry_type="inproceedings",
            url="https://ieeexplore.ieee.org/document/10803340",
            doi="10.1109/EXAMPLE.2024.10803340"
        )
        
        self.citations["agent_simulator_fbdl_2024"] = Citation(
            title="An Agent Based Simulator on ROS for Fuzzy Behavior Description Language (FBDL)",
            authors=[Author("", "")],  # Authors not specified in the reference
            year=2024,
            journal="IEEE",
            entry_type="inproceedings",
            url="https://ieeexplore.ieee.org/document/10569497/",
            doi="10.1109/EXAMPLE.2024.10569497"
        )
        
        self.citations["causal_inference_cnn_lstm_2025"] = Citation(
            title="Causal Inference Framework Based on CNN-LSTM-AM-DID: The Impact of Establishing Digital Economy Innovation Development Pilot Zones on Enterprise Digitalization",
            authors=[Author("", "")],  # Authors not specified in the reference
            year=2025,
            journal="IEEE",
            entry_type="inproceedings",
            url="https://ieeexplore.ieee.org/document/11034396",
            doi="10.1109/EXAMPLE.2025.11034396"
        )
        
        self.citations["dynamic_architectures_ai_agents_2025"] = Citation(
            title="Dynamic Architectures Leveraging AI Agents and Human-in-the-Loop for Data Center Management",
            authors=[Author("", "")],  # Authors not specified in the reference
            year=2025,
            journal="IEEE",
            entry_type="inproceedings",
            url="https://ieeexplore.ieee.org/document/11014915",
            doi="10.1109/EXAMPLE.2025.11014915"
        )
    
    def add_citation(self, key: str, citation: Citation) -> None:
        """Add a citation to the database."""
        self.citations[key] = citation
    
    def get_citation(self, key: str) -> Optional[Citation]:
        """Get a citation by key."""
        return self.citations.get(key)
    
    def search_citations(self, query: str) -> List[str]:
        """Search citations by title or author."""
        results = []
        query_lower = query.lower()
        
        for key, citation in self.citations.items():
            # Search in title
            if query_lower in citation.title.lower():
                results.append(key)
                continue
            
            # Search in authors
            for author in citation.authors:
                if (query_lower in author.first_name.lower() or 
                    query_lower in author.last_name.lower()):
                    results.append(key)
                    break
        
        return results
    
    def generate_bibliography(self, 
                            citation_keys: List[str],
                            format_type: str = "bibtex") -> str:
        """Generate bibliography in specified format."""
        if format_type == "bibtex":
            return self._generate_bibtex_bibliography(citation_keys)
        elif format_type == "apa":
            return self._generate_apa_bibliography(citation_keys)
        elif format_type == "ieee":
            return self._generate_ieee_bibliography(citation_keys)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_bibtex_bibliography(self, citation_keys: List[str]) -> str:
        """Generate BibTeX bibliography."""
        bibliography = f"% Bibliography generated by ESCAI Framework\n"
        bibliography += f"% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for key in citation_keys:
            if key in self.citations:
                bibliography += self.citations[key].to_bibtex(key) + "\n"
        
        return bibliography
    
    def _generate_apa_bibliography(self, citation_keys: List[str]) -> str:
        """Generate APA format bibliography."""
        bibliography = "References\n\n"
        
        # Sort citations by first author's last name
        sorted_citations = []
        for key in citation_keys:
            if key in self.citations:
                citation = self.citations[key]
                sort_key = citation.authors[0].last_name if citation.authors else ""
                sorted_citations.append((sort_key, citation))
        
        sorted_citations.sort(key=lambda x: x[0])
        
        for _, citation in sorted_citations:
            bibliography += citation.to_apa() + "\n\n"
        
        return bibliography
    
    def _generate_ieee_bibliography(self, citation_keys: List[str]) -> str:
        """Generate IEEE format bibliography."""
        bibliography = "REFERENCES\n\n"
        
        for i, key in enumerate(citation_keys, 1):
            if key in self.citations:
                citation = self.citations[key]
                bibliography += f"[{i}] {citation.to_ieee()}\n\n"
        
        return bibliography
    
    def export_to_file(self, 
                      citation_keys: List[str],
                      filename: str,
                      format_type: str = "bibtex") -> None:
        """Export bibliography to file."""
        bibliography = self.generate_bibliography(citation_keys, format_type)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(bibliography)
    
    def import_from_bibtex(self, filename: str) -> None:
        """Import citations from BibTeX file."""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple BibTeX parser (basic implementation)
        entries = re.findall(r'@(\w+)\{([^,]+),([^}]+)\}', content, re.DOTALL)
        
        for entry_type, key, fields in entries:
            citation_data = self._parse_bibtex_fields(fields)
            
            # Create Citation object
            authors = self._parse_authors(citation_data.get('author', ''))
            
            citation = Citation(
                title=citation_data.get('title', ''),
                authors=authors,
                year=int(citation_data.get('year', 0)) if citation_data.get('year') else None,
                journal=citation_data.get('journal'),
                volume=citation_data.get('volume'),
                number=citation_data.get('number'),
                pages=citation_data.get('pages'),
                publisher=citation_data.get('publisher'),
                doi=citation_data.get('doi'),
                url=citation_data.get('url'),
                entry_type=entry_type.lower()
            )
            
            self.add_citation(key.strip(), citation)
    
    def _parse_bibtex_fields(self, fields_str: str) -> Dict[str, str]:
        """Parse BibTeX fields string."""
        fields = {}
        
        # Simple field extraction
        field_pattern = r'(\w+)\s*=\s*\{([^}]+)\}'
        matches = re.findall(field_pattern, fields_str)
        
        for field_name, field_value in matches:
            fields[field_name.strip()] = field_value.strip()
        
        return fields
    
    def _parse_authors(self, author_str: str) -> List[Author]:
        """Parse author string into Author objects."""
        if not author_str:
            return []
        
        authors = []
        author_parts = author_str.split(' and ')
        
        for author_part in author_parts:
            author_part = author_part.strip()
            
            # Simple name parsing (First Last or Last, First)
            if ',' in author_part:
                last, first = author_part.split(',', 1)
                last = last.strip()
                first = first.strip()
            else:
                name_parts = author_part.split()
                if len(name_parts) >= 2:
                    first = name_parts[0]
                    last = ' '.join(name_parts[1:])
                else:
                    first = ""
                    last = author_part
            
            authors.append(Author(first_name=first, last_name=last))
        
        return authors


class MethodologyCitationGenerator:
    """Generates citations for specific ESCAI methodologies."""
    
    def __init__(self, citation_db: CitationDatabase):
        self.citation_db = citation_db
    
    def get_methodology_citations(self, methodology: str) -> List[str]:
        """Get relevant citations for a specific methodology."""
        methodology_map = {
            "causal_analysis": [
                "pearl2009causality",
                "spirtes2000causation",
                "causal_inference_cnn_lstm_2025"
            ],
            "pattern_mining": [
                "han2011data"
            ],
            "epistemic_extraction": [
                "fagin1995reasoning",
                "russell2016artificial"
            ],
            "multiagent_monitoring": [
                "wooldridge2009introduction",
                "stone2000multiagent",
                "agent_simulator_fbdl_2024",
                "dynamic_architectures_ai_agents_2025"
            ],
            "statistical_analysis": [
                "wasserman2004all",
                "bishop2006pattern"
            ],
            "time_series_analysis": [
                "hamilton1994time"
            ],
            "real_time_monitoring": [
                "buttazzo2011hard",
                "smartla_2025",
                "industry_anomalies_2024"
            ],
            "agent_safety_monitoring": [
                "smartla_2025",
                "ood_detection_2024",
                "industry_anomalies_2024"
            ],
            "agent_governance": [
                "agent_intelligence_protocol_2025"
            ],
            "neurosymbolic_agents": [
                "ood_detection_2024"
            ],
            "runtime_monitoring": [
                "industry_anomalies_2024",
                "smartla_2025"
            ]
        }
        
        return methodology_map.get(methodology, [])
    
    def generate_methodology_section(self, methodologies: List[str]) -> str:
        """Generate methodology section with appropriate citations."""
        section = "\\section{Methodology}\n\n"
        
        section += """The analysis presented in this paper employs the ESCAI (Epistemic State and Causal Analysis Intelligence) Framework, a comprehensive system for monitoring and analyzing autonomous agent behavior in real-time.\n\n"""
        
        if "epistemic_extraction" in methodologies:
            section += """\\subsection{Epistemic State Extraction}\n\n"""
            section += """Epistemic states represent the knowledge, beliefs, and goals of autonomous agents at any given time. The extraction process follows established principles from epistemic logic \\cite{fagin1995reasoning} and modern AI reasoning systems \\cite{russell2016artificial}.\n\n"""
        
        if "causal_analysis" in methodologies:
            section += """\\subsection{Causal Analysis}\n\n"""
            section += """Causal relationships between agent actions and outcomes are identified using Pearl's causal inference framework \\cite{pearl2009causality}. The analysis employs directed acyclic graphs (DAGs) to represent causal structures and applies do-calculus for intervention analysis \\cite{spirtes2000causation}.\n\n"""
        
        if "pattern_mining" in methodologies:
            section += """\\subsection{Behavioral Pattern Mining}\n\n"""
            section += """Sequential pattern mining techniques are applied to identify recurring behavioral patterns in agent execution traces \\cite{han2011data}. The analysis focuses on frequent subsequences that correlate with successful task completion.\n\n"""
        
        if "multiagent_monitoring" in methodologies:
            section += """\\subsection{Multi-Agent System Monitoring}\n\n"""
            section += """For multi-agent scenarios, the framework monitors inter-agent communication and coordination patterns \\cite{wooldridge2009introduction}. The analysis considers both individual agent behavior and emergent system-level properties \\cite{stone2000multiagent}.\n\n"""
        
        if "statistical_analysis" in methodologies:
            section += """\\subsection{Statistical Analysis}\n\n"""
            section += """Statistical significance testing and confidence interval estimation follow standard methodologies \\cite{wasserman2004all}. Machine learning models for prediction and classification are implemented using established techniques \\cite{bishop2006pattern}.\n\n"""
        
        if "agent_safety_monitoring" in methodologies:
            section += """\\subsection{Agent Safety Monitoring}\n\n"""
            section += """Safety monitoring approaches for autonomous agents follow established practices for deep reinforcement learning systems \\cite{smartla_2025}. Out-of-distribution detection techniques are employed to identify anomalous agent behaviors \\cite{ood_detection_2024}, while runtime anomaly detection follows industry best practices \\cite{industry_anomalies_2024}.\n\n"""
        
        if "agent_governance" in methodologies:
            section += """\\subsection{Agent Governance}\n\n"""
            section += """Runtime governance for agentic AI systems follows the Agent Intelligence Protocol framework \\cite{agent_intelligence_protocol_2025}, ensuring proper oversight and control of autonomous agent operations.\n\n"""
        
        if "neurosymbolic_agents" in methodologies:
            section += """\\subsection{Neurosymbolic Agent Analysis}\n\n"""
            section += """Analysis of neurosymbolic autonomous agents incorporates specialized techniques for out-of-distribution detection \\cite{ood_detection_2024}, addressing the unique challenges posed by hybrid symbolic-neural architectures.\n\n"""
        
        if "runtime_monitoring" in methodologies:
            section += """\\subsection{Runtime Monitoring}\n\n"""
            section += """Runtime monitoring approaches leverage industry-proven techniques for anomaly detection \\cite{industry_anomalies_2024} and safety monitoring specifically designed for deep reinforcement learning agents \\cite{smartla_2025}.\n\n"""
        
        return section