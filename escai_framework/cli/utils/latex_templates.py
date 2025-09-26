"""
LaTeX template system for academic publications.

Provides specialized templates for different academic venues and publication types.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LatexTemplate:
    """Represents a LaTeX document template."""
    name: str
    document_class: str
    packages: List[str]
    preamble: str
    title_format: str
    author_format: str
    abstract_format: str
    content_structure: List[str]
    bibliography_style: str


class LatexTemplateManager:
    """Manages LaTeX templates for different publication venues."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, LatexTemplate]:
        """Load predefined LaTeX templates."""
        templates = {}
        
        # IEEE Conference Template
        templates['ieee_conference'] = LatexTemplate(
            name="IEEE Conference",
            document_class="\\documentclass[conference]{IEEEtran}",
            packages=[
                "\\usepackage{cite}",
                "\\usepackage{amsmath,amssymb,amsfonts}",
                "\\usepackage{algorithmic}",
                "\\usepackage{graphicx}",
                "\\usepackage{textcomp}",
                "\\usepackage{xcolor}",
                "\\usepackage{booktabs}",
                "\\usepackage{multirow}",
                "\\usepackage{array}"
            ],
            preamble="",
            title_format="\\title{{{title}}}",
            author_format="\\author{{\n{authors}\n}}",
            abstract_format="\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}",
            content_structure=[
                "\\maketitle",
                "{abstract}",
                "\\begin{IEEEkeywords}",
                "{keywords}",
                "\\end{IEEEkeywords}",
                "{content}"
            ],
            bibliography_style="IEEEtran"
        )
        
        # ACM Template
        templates['acm'] = LatexTemplate(
            name="ACM",
            document_class="\\documentclass[sigconf]{acmart}",
            packages=[
                "\\usepackage{booktabs}",
                "\\usepackage{multirow}",
                "\\usepackage{array}"
            ],
            preamble="\\settopmatter{printacmref=false}",
            title_format="\\title{{{title}}}",
            author_format="{authors}",
            abstract_format="\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}",
            content_structure=[
                "{title}",
                "{authors}",
                "{abstract}",
                "\\keywords{{{keywords}}}",
                "\\maketitle",
                "{content}"
            ],
            bibliography_style="ACM-Reference-Format"
        )
        
        # Springer LNCS Template
        templates['springer_lncs'] = LatexTemplate(
            name="Springer LNCS",
            document_class="\\documentclass{llncs}",
            packages=[
                "\\usepackage{graphicx}",
                "\\usepackage{booktabs}",
                "\\usepackage{multirow}",
                "\\usepackage{array}",
                "\\usepackage{amsmath}"
            ],
            preamble="",
            title_format="\\title{{{title}}}",
            author_format="\\author{{{authors}}}",
            abstract_format="\\begin{{abstract}}\n{abstract}\n\\keywords{{{keywords}}}\n\\end{{abstract}}",
            content_structure=[
                "{title}",
                "{authors}",
                "\\institute{{Your Institution}}",
                "\\maketitle",
                "{abstract}",
                "{content}"
            ],
            bibliography_style="splncs04"
        )
        
        # Elsevier Template
        templates['elsevier'] = LatexTemplate(
            name="Elsevier",
            document_class="\\documentclass[review]{elsarticle}",
            packages=[
                "\\usepackage{lineno,hyperref}",
                "\\usepackage{booktabs}",
                "\\usepackage{multirow}",
                "\\usepackage{array}",
                "\\usepackage{amsmath}"
            ],
            preamble="\\modulolinenumbers[5]",
            title_format="\\title{{{title}}}",
            author_format="{authors}",
            abstract_format="\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}",
            content_structure=[
                "\\begin{frontmatter}",
                "{title}",
                "{authors}",
                "{abstract}",
                "\\begin{keyword}",
                "{keywords}",
                "\\end{keyword}",
                "\\end{frontmatter}",
                "\\linenumbers",
                "{content}"
            ],
            bibliography_style="elsarticle-num"
        )
        
        # Generic Academic Template
        templates['generic'] = LatexTemplate(
            name="Generic Academic",
            document_class="\\documentclass[12pt,a4paper]{article}",
            packages=[
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage{graphicx}",
                "\\usepackage{amsmath}",
                "\\usepackage{cite}",
                "\\usepackage{booktabs}",
                "\\usepackage{multirow}",
                "\\usepackage{array}",
                "\\usepackage[margin=1in]{geometry}"
            ],
            preamble="",
            title_format="\\title{{{title}}}",
            author_format="\\author{{{authors}}}",
            abstract_format="\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}",
            content_structure=[
                "{title}",
                "{authors}",
                "\\date{\\today}",
                "\\maketitle",
                "{abstract}",
                "{content}"
            ],
            bibliography_style="plain"
        )
        
        return templates
    
    def get_template(self, template_name: str) -> LatexTemplate:
        """Get a specific template by name."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[template_name]
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())
    
    def generate_document(self, 
                         template_name: str,
                         title: str,
                         authors: List[str],
                         abstract: str,
                         content: str,
                         keywords: Optional[List[str]] = None,
                         bibliography_file: str = "references") -> str:
        """Generate complete LaTeX document from template."""
        template = self.get_template(template_name)
        
        # Format authors based on template
        formatted_authors = self._format_authors(authors, template_name)
        
        # Format keywords
        keywords_str = ", ".join(keywords) if keywords else "ESCAI, agent monitoring, epistemic states"
        
        # Build document
        document = template.document_class + "\n"
        
        # Add packages
        for package in template.packages:
            document += package + "\n"
        
        # Add preamble
        if template.preamble:
            document += "\n" + template.preamble + "\n"
        
        document += "\n\\begin{document}\n\n"
        
        # Add title and authors before content structure for templates that don't include them
        if template_name in ['ieee_conference', 'acm', 'springer_lncs', 'elsevier']:
            document += template.title_format.format(title=title) + "\n\n"
            document += template.author_format.format(authors=formatted_authors) + "\n\n"
        
        # Build content structure
        for element in template.content_structure:
            if element == "{title}":
                document += template.title_format.format(title=title) + "\n\n"
            elif element == "{authors}":
                document += template.author_format.format(authors=formatted_authors) + "\n\n"
            elif element == "{abstract}":
                document += template.abstract_format.format(abstract=abstract, keywords=keywords_str) + "\n\n"
            elif element == "{keywords}":
                document += keywords_str
            elif element == "{content}":
                document += content + "\n\n"
            else:
                document += element + "\n"
        
        # Add bibliography
        document += f"\\bibliographystyle{{{template.bibliography_style}}}\n"
        document += f"\\bibliography{{{bibliography_file}}}\n\n"
        
        document += "\\end{document}\n"
        
        return document
    
    def _format_authors(self, authors: List[str], template_name: str) -> str:
        """Format authors according to template requirements."""
        if template_name == 'ieee_conference':
            formatted = []
            for i, author in enumerate(authors):
                formatted.append(f"\\IEEEauthorblockN{{{author}}}")
            return "\\\\\n".join(formatted)
        
        elif template_name == 'acm':
            formatted = []
            for author in authors:
                formatted.append(f"\\author{{{author}}}")
            return "\n".join(formatted)
        
        elif template_name == 'springer_lncs':
            return " \\and ".join(authors)
        
        elif template_name == 'elsevier':
            formatted = []
            for author in authors:
                formatted.append(f"\\author{{{author}}}")
            return "\n".join(formatted)
        
        else:  # generic
            return " \\\\ ".join(authors)


class LatexTableGenerator:
    """Generates LaTeX tables with proper formatting."""
    
    @staticmethod
    def generate_results_table(data: Dict[str, Any], 
                             caption: str,
                             label: str,
                             position: str = "htbp") -> str:
        """Generate a results table in LaTeX format."""
        if isinstance(data, dict):
            return LatexTableGenerator._dict_to_table(data, caption, label, position)
        else:
            raise ValueError("Data must be a dictionary")
    
    @staticmethod
    def _dict_to_table(data: Dict[str, Any], 
                      caption: str,
                      label: str,
                      position: str) -> str:
        """Convert dictionary to LaTeX table."""
        # Determine table structure
        num_cols = 2  # Key-Value pairs
        col_spec = "ll"
        
        latex = f"\\begin{{table}}[{position}]\n"
        latex += "\\centering\n"
        latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex += "\\toprule\n"
        
        # Header
        latex += "Metric & Value \\\\\n"
        latex += "\\midrule\n"
        
        # Data rows
        for key, value in data.items():
            key_formatted = key.replace('_', ' ').title()
            value_formatted = LatexTableGenerator._format_value(value)
            latex += f"{key_formatted} & {value_formatted} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    @staticmethod
    def generate_comparison_table(data: Dict[str, Dict[str, Any]], 
                                caption: str,
                                label: str,
                                position: str = "htbp") -> str:
        """Generate comparison table for multiple datasets."""
        if not data:
            return ""
        
        # Get all unique metrics
        all_metrics: set[str] = set()
        for dataset_data in data.values():
            all_metrics.update(dataset_data.keys())
        
        metrics = sorted(list(all_metrics))
        datasets = list(data.keys())
        
        # Table structure
        num_cols = len(datasets) + 1
        col_spec = "l" + "c" * len(datasets)
        
        latex = f"\\begin{{table}}[{position}]\n"
        latex += "\\centering\n"
        latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex += "\\toprule\n"
        
        # Header
        header = "Metric & " + " & ".join(datasets) + " \\\\\n"
        latex += header
        latex += "\\midrule\n"
        
        # Data rows
        for metric in metrics:
            row = metric.replace('_', ' ').title()
            for dataset in datasets:
                value = data[dataset].get(metric, "N/A")
                formatted_value = LatexTableGenerator._format_value(value)
                row += f" & {formatted_value}"
            row += " \\\\\n"
            latex += row
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    @staticmethod
    def _format_value(value: Any) -> str:
        """Format value for LaTeX table."""
        if isinstance(value, float):
            if value < 0.001:
                return f"{value:.2e}"
            else:
                return f"{value:.3f}"
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, str):
            # Escape LaTeX special characters
            value = value.replace('&', '\\&')
            value = value.replace('%', '\\%')
            value = value.replace('$', '\\$')
            value = value.replace('#', '\\#')
            value = value.replace('_', '\\_')
            return value
        else:
            return str(value)


class LatexFigureGenerator:
    """Generates LaTeX figures and plots."""
    
    @staticmethod
    def generate_figure_environment(figure_path: str,
                                  caption: str,
                                  label: str,
                                  width: str = "0.8\\textwidth",
                                  position: str = "htbp") -> str:
        """Generate LaTeX figure environment."""
        latex = f"\\begin{{figure}}[{position}]\n"
        latex += "\\centering\n"
        latex += f"\\includegraphics[width={width}]{{{figure_path}}}\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\end{figure}\n"
        
        return latex
    
    @staticmethod
    def generate_subfigures(figures: List[Dict[str, str]],
                          main_caption: str,
                          main_label: str,
                          position: str = "htbp") -> str:
        """Generate subfigures environment."""
        latex = f"\\begin{{figure}}[{position}]\n"
        latex += "\\centering\n"
        
        for i, fig in enumerate(figures):
            width = fig.get('width', '0.45\\textwidth')
            latex += f"\\begin{{subfigure}}{{{width}}}\n"
            latex += "\\centering\n"
            latex += f"\\includegraphics[width=\\textwidth]{{{fig['path']}}}\n"
            latex += f"\\caption{{{fig['caption']}}}\n"
            latex += f"\\label{{{fig['label']}}}\n"
            latex += "\\end{subfigure}\n"
            
            if i < len(figures) - 1:
                latex += "\\hfill\n"
        
        latex += f"\\caption{{{main_caption}}}\n"
        latex += f"\\label{{{main_label}}}\n"
        latex += "\\end{figure}\n"
        
        return latex