"""
Publication-ready output generation for ESCAI CLI.

This module provides comprehensive formatting capabilities for academic publications,
including LaTeX output, citation generation, and statistical report formatting.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


@dataclass
class Citation:
    """Represents a citation for methodology references."""
    authors: List[str]
    title: str
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pages: Optional[str] = None
    volume: Optional[str] = None
    
    def to_bibtex(self, key: str) -> str:
        """Generate BibTeX entry for this citation."""
        entry_type = "article" if self.journal else "misc"
        
        bibtex = f"@{entry_type}{{{key},\n"
        bibtex += f"  title={{{self.title}}},\n"
        bibtex += f"  author={{{' and '.join(self.authors)}}},\n"
        
        if self.journal:
            bibtex += f"  journal={{{self.journal}}},\n"
        if self.year:
            bibtex += f"  year={{{self.year}}},\n"
        if self.volume:
            bibtex += f"  volume={{{self.volume}}},\n"
        if self.pages:
            bibtex += f"  pages={{{self.pages}}},\n"
        if self.doi:
            bibtex += f"  doi={{{self.doi}}},\n"
        if self.url:
            bibtex += f"  url={{{self.url}}},\n"
            
        bibtex += "}\n"
        return bibtex
    
    def to_apa(self) -> str:
        """Generate APA format citation."""
        authors_str = ", ".join(self.authors)
        if len(self.authors) > 1:
            authors_str = authors_str.rsplit(", ", 1)
            authors_str = f"{authors_str[0]}, & {authors_str[1]}"
        
        citation = f"{authors_str}"
        if self.year:
            citation += f" ({self.year})"
        citation += f". {self.title}"
        
        if self.journal:
            citation += f". *{self.journal}*"
            if self.volume:
                citation += f", {self.volume}"
            if self.pages:
                citation += f", {self.pages}"
        
        if self.doi:
            citation += f". https://doi.org/{self.doi}"
        elif self.url:
            citation += f". {self.url}"
            
        return citation + "."


@dataclass
class Figure:
    """Represents a figure for publication."""
    id: str
    title: str
    caption: str
    data: Any
    figure_type: str  # 'table', 'chart', 'plot'
    width: Optional[str] = None
    height: Optional[str] = None
    
    def to_latex(self) -> str:
        """Generate LaTeX figure environment."""
        latex = "\\begin{figure}[htbp]\n"
        latex += "\\centering\n"
        
        if self.figure_type == 'table':
            latex += self._generate_latex_table()
        else:
            latex += f"% Insert {self.figure_type} here\n"
            latex += f"% Data: {self.data}\n"
        
        latex += f"\\caption{{{self.caption}}}\n"
        latex += f"\\label{{fig:{self.id}}}\n"
        latex += "\\end{figure}\n"
        
        return latex
    
    def _generate_latex_table(self) -> str:
        """Generate LaTeX table from data."""
        if isinstance(self.data, pd.DataFrame):
            return self._dataframe_to_latex(self.data)
        elif isinstance(self.data, dict):
            return self._dict_to_latex(self.data)
        else:
            return f"% Unsupported data type: {type(self.data)}\n"
    
    def _dataframe_to_latex(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to LaTeX table."""
        num_cols = len(df.columns)
        col_spec = "l" + "c" * (num_cols - 1)
        
        latex = f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex += "\\hline\n"
        
        # Header
        headers = " & ".join([self._escape_latex(str(col)) for col in df.columns])
        latex += f"{headers} \\\\\n"
        latex += "\\hline\n"
        
        # Data rows
        for _, row in df.iterrows():
            row_data = " & ".join([self._escape_latex(str(val)) for val in row])
            latex += f"{row_data} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        
        return latex
    
    def _dict_to_latex(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to LaTeX table."""
        latex = "\\begin{tabular}{ll}\n"
        latex += "\\hline\n"
        
        for key, value in data.items():
            key_escaped = self._escape_latex(str(key))
            value_escaped = self._escape_latex(str(value))
            latex += f"{key_escaped} & {value_escaped} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        
        return latex
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '^': '\\textasciicircum{}',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '\\': '\\textbackslash{}'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text


class PublicationFormatter:
    """Main class for generating publication-ready outputs."""
    
    def __init__(self):
        self.console = Console()
        self.citations: Dict[str, Citation] = {}
        self.figures: List[Figure] = []
        self.bibliography = Bibliography()
        
    def add_citation(self, key: str, citation: Citation) -> None:
        """Add a citation to the bibliography."""
        self.citations[key] = citation
        
    def add_figure(self, figure: Figure) -> None:
        """Add a figure to the document."""
        self.figures.append(figure)
        
    def generate_academic_template(self, 
                                 title: str,
                                 authors: List[str],
                                 abstract: str,
                                 content: str,
                                 template_type: str = "ieee") -> str:
        """Generate academic paper template."""
        if template_type == "ieee":
            return self._generate_ieee_template(title, authors, abstract, content)
        elif template_type == "acm":
            return self._generate_acm_template(title, authors, abstract, content)
        elif template_type == "springer":
            return self._generate_springer_template(title, authors, abstract, content)
        else:
            return self._generate_generic_template(title, authors, abstract, content)
    
    def _generate_ieee_template(self, title: str, authors: List[str], 
                               abstract: str, content: str) -> str:
        """Generate IEEE conference paper template."""
        template = """\\documentclass[conference]{IEEEtran}
\\usepackage{cite}
\\usepackage{amsmath,amssymb,amsfonts}
\\usepackage{algorithmic}
\\usepackage{graphicx}
\\usepackage{textcomp}
\\usepackage{xcolor}

\\begin{document}

\\title{""" + title + """}

\\author{
""" + "\\\\".join([f"\\IEEEauthorblockN{{{author}}}" for author in authors]) + """
}

\\maketitle

\\begin{abstract}
""" + abstract + """
\\end{abstract}

\\begin{IEEEkeywords}
ESCAI, agent monitoring, epistemic states, causal analysis
\\end{IEEEkeywords}

""" + content + """

\\bibliographystyle{IEEEtran}
\\bibliography{references}

\\end{document}"""
        
        return template
    
    def _generate_acm_template(self, title: str, authors: List[str], 
                              abstract: str, content: str) -> str:
        """Generate ACM paper template."""
        template = """\\documentclass[sigconf]{acmart}

\\begin{document}

\\title{""" + title + """}

""" + "\n".join([f"\\author{{{author}}}" for author in authors]) + """

\\begin{abstract}
""" + abstract + """
\\end{abstract}

\\keywords{ESCAI, agent monitoring, epistemic states, causal analysis}

\\maketitle

""" + content + """

\\bibliographystyle{ACM-Reference-Format}
\\bibliography{references}

\\end{document}"""
        
        return template
    
    def _generate_springer_template(self, title: str, authors: List[str], 
                                   abstract: str, content: str) -> str:
        """Generate Springer LNCS template."""
        template = """\\documentclass{llncs}
\\usepackage{graphicx}

\\begin{document}

\\title{""" + title + """}

\\author{""" + " \\and ".join(authors) + """}

\\institute{Your Institution}

\\maketitle

\\begin{abstract}
""" + abstract + """
\\keywords{ESCAI \\and agent monitoring \\and epistemic states \\and causal analysis}
\\end{abstract}

""" + content + """

\\bibliographystyle{splncs04}
\\bibliography{references}

\\end{document}"""
        
        return template
    
    def _generate_generic_template(self, title: str, authors: List[str], 
                                  abstract: str, content: str) -> str:
        """Generate generic academic template."""
        template = """\\documentclass[12pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
\\usepackage{amsmath}
\\usepackage{cite}

\\title{""" + title + """}
\\author{""" + " \\\\ ".join(authors) + """}
\\date{\\today}

\\begin{document}

\\maketitle

\\begin{abstract}
""" + abstract + """
\\end{abstract}

""" + content + """

\\bibliographystyle{plain}
\\bibliography{references}

\\end{document}"""
        
        return template


class Bibliography:
    """Manages bibliography and citation generation."""
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}
        self._load_default_citations()
    
    def _load_default_citations(self) -> None:
        """Load default citations for ESCAI methodologies."""
        # Add common citations for agent monitoring and epistemic analysis
        self.citations["pearl2009causality"] = Citation(
            authors=["Pearl, Judea"],
            title="Causality: Models, Reasoning and Inference",
            year=2009,
            journal="Cambridge University Press"
        )
        
        self.citations["russell2016artificial"] = Citation(
            authors=["Russell, Stuart", "Norvig, Peter"],
            title="Artificial Intelligence: A Modern Approach",
            year=2016,
            journal="Pearson"
        )
        
        self.citations["wooldridge2009introduction"] = Citation(
            authors=["Wooldridge, Michael"],
            title="An Introduction to MultiAgent Systems",
            year=2009,
            journal="John Wiley & Sons"
        )
    
    def add_citation(self, key: str, citation: Citation) -> None:
        """Add a citation to the bibliography."""
        self.citations[key] = citation
    
    def generate_bibtex(self) -> str:
        """Generate complete BibTeX bibliography."""
        bibtex = "% Generated bibliography for ESCAI Framework\n"
        bibtex += f"% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for key, citation in self.citations.items():
            bibtex += citation.to_bibtex(key) + "\n"
        
        return bibtex
    
    def generate_apa_bibliography(self) -> str:
        """Generate APA format bibliography."""
        bibliography = "References\n\n"
        
        sorted_citations = sorted(self.citations.items(), 
                                key=lambda x: x[1].authors[0] if x[1].authors else "")
        
        for key, citation in sorted_citations:
            bibliography += citation.to_apa() + "\n\n"
        
        return bibliography


class StatisticalReportGenerator:
    """Generates statistical reports with proper academic formatting."""
    
    def __init__(self, formatter: PublicationFormatter):
        self.formatter = formatter
    
    def generate_analysis_report(self, 
                               analysis_results: Dict[str, Any],
                               methodology: str = "ESCAI Framework") -> str:
        """Generate comprehensive statistical analysis report."""
        report = self._generate_methodology_section(methodology)
        report += self._generate_results_section(analysis_results)
        report += self._generate_discussion_section(analysis_results)
        
        return report
    
    def _generate_methodology_section(self, methodology: str) -> str:
        """Generate methodology section with proper citations."""
        section = "\\section{Methodology}\n\n"
        
        if methodology == "ESCAI Framework":
            section += """The analysis was conducted using the ESCAI (Epistemic State and Causal Analysis Intelligence) Framework, which provides real-time monitoring and analysis of autonomous agent behavior. The framework employs a multi-layered approach to extract epistemic states, identify behavioral patterns, and perform causal inference \\cite{pearl2009causality}.

The monitoring process captures agent execution events in real-time, extracting key epistemic indicators including beliefs, goals, and knowledge states. Statistical analysis is performed using established methods for time-series analysis and pattern recognition \\cite{russell2016artificial}.

"""
        
        return section
    
    def _generate_results_section(self, results: Dict[str, Any]) -> str:
        """Generate results section with tables and figures."""
        section = "\\section{Results}\n\n"
        
        # Generate summary statistics table
        if 'summary_stats' in results:
            stats_table = self._create_summary_stats_table(results['summary_stats'])
            self.formatter.add_figure(stats_table)
            section += f"Table \\ref{{fig:{stats_table.id}}} presents the summary statistics for the monitored agents.\n\n"
        
        # Generate performance metrics
        if 'performance_metrics' in results:
            perf_table = self._create_performance_table(results['performance_metrics'])
            self.formatter.add_figure(perf_table)
            section += f"Performance metrics are detailed in Table \\ref{{fig:{perf_table.id}}}.\n\n"
        
        return section
    
    def _generate_discussion_section(self, results: Dict[str, Any]) -> str:
        """Generate discussion section with interpretation."""
        section = "\\section{Discussion}\n\n"
        
        section += """The results demonstrate the effectiveness of real-time epistemic monitoring in understanding agent behavior. The observed patterns indicate significant correlations between epistemic state transitions and task performance outcomes.

Key findings include:
\\begin{itemize}
\\item Epistemic uncertainty correlates with task completion time
\\item Goal revision frequency impacts overall success rates  
\\item Belief consistency serves as a predictor of agent reliability
\\end{itemize}

These findings have important implications for agent system design and monitoring strategies.

"""
        
        return section
    
    def _create_summary_stats_table(self, stats: Dict[str, Any]) -> Figure:
        """Create summary statistics table."""
        df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        df.index.name = 'Metric'
        
        return Figure(
            id="summary_stats",
            title="Summary Statistics",
            caption="Summary statistics for monitored agent sessions",
            data=df,
            figure_type="table"
        )
    
    def _create_performance_table(self, metrics: Dict[str, Any]) -> Figure:
        """Create performance metrics table."""
        df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
        df.index.name = 'Performance Metric'
        
        return Figure(
            id="performance_metrics", 
            title="Performance Metrics",
            caption="Performance evaluation metrics for agent monitoring",
            data=df,
            figure_type="table"
        )


def format_for_publication(data: Dict[str, Any], 
                         output_format: str = "latex",
                         template_type: str = "ieee") -> str:
    """Main function to format data for publication."""
    formatter = PublicationFormatter()
    
    # Add default citations
    formatter.bibliography._load_default_citations()
    
    if output_format == "latex":
        return _generate_latex_output(data, formatter, template_type)
    elif output_format == "markdown":
        return _generate_markdown_output(data, formatter)
    elif output_format == "html":
        return _generate_html_output(data, formatter)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _generate_latex_output(data: Dict[str, Any], 
                          formatter: PublicationFormatter,
                          template_type: str) -> str:
    """Generate LaTeX formatted output."""
    title = data.get('title', 'ESCAI Framework Analysis Results')
    authors = data.get('authors', ['Research Team'])
    abstract = data.get('abstract', 'Analysis results from ESCAI Framework monitoring.')
    
    # Generate content sections
    content = "\\section{Introduction}\n\n"
    content += "This report presents analysis results from the ESCAI Framework.\n\n"
    
    # Add statistical report if available
    if 'analysis_results' in data:
        report_gen = StatisticalReportGenerator(formatter)
        content += report_gen.generate_analysis_report(data['analysis_results'])
    
    # Add figures
    for figure in formatter.figures:
        content += figure.to_latex() + "\n"
    
    # Generate complete document using LaTeX template manager
    from .latex_templates import LatexTemplateManager
    template_manager = LatexTemplateManager()
    
    document = template_manager.generate_document(
        template_type, title, authors, abstract, content
    )
    
    return document


def _generate_markdown_output(data: Dict[str, Any], 
                             formatter: PublicationFormatter) -> str:
    """Generate Markdown formatted output."""
    output = f"# {data.get('title', 'ESCAI Framework Analysis Results')}\n\n"
    
    authors = data.get('authors', ['Research Team'])
    output += f"**Authors:** {', '.join(authors)}\n\n"
    
    output += f"**Abstract:** {data.get('abstract', 'Analysis results from ESCAI Framework monitoring.')}\n\n"
    
    output += "## Results\n\n"
    
    if 'analysis_results' in data:
        results = data['analysis_results']
        for key, value in results.items():
            output += f"### {key.replace('_', ' ').title()}\n\n"
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    output += f"- **{subkey}:** {subvalue}\n"
            else:
                output += f"{value}\n"
            output += "\n"
    
    return output


def _generate_html_output(data: Dict[str, Any], 
                         formatter: PublicationFormatter) -> str:
    """Generate HTML formatted output."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{data.get('title', 'ESCAI Framework Analysis Results')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{data.get('title', 'ESCAI Framework Analysis Results')}</h1>
    
    <p><strong>Authors:</strong> {', '.join(data.get('authors', ['Research Team']))}</p>
    
    <h2>Abstract</h2>
    <p>{data.get('abstract', 'Analysis results from ESCAI Framework monitoring.')}</p>
    
    <h2>Results</h2>
"""
    
    if 'analysis_results' in data:
        results = data['analysis_results']
        for key, value in results.items():
            html += f"<h3>{key.replace('_', ' ').title()}</h3>\n"
            if isinstance(value, dict):
                html += "<ul>\n"
                for subkey, subvalue in value.items():
                    html += f"<li><strong>{subkey}:</strong> {subvalue}</li>\n"
                html += "</ul>\n"
            else:
                html += f"<p>{value}</p>\n"
    
    html += """
</body>
</html>"""
    
    return html