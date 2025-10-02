"""
Publication-ready output generation commands for ESCAI CLI.

Provides commands for generating academic papers, reports, and citations
from ESCAI monitoring and analysis data.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.publication_formatter import (
    PublicationFormatter, 
    format_for_publication,
    StatisticalReportGenerator as BaseStatisticalReportGenerator
)
from ..utils.latex_templates import LatexTemplateManager, LatexTableGenerator
from ..utils.citation_manager import CitationDatabase, MethodologyCitationGenerator
from ..utils.statistical_report_generator import StatisticalReportGenerator
# from ..utils.formatters import format_output  # Not needed for publication commands


console = Console()


@click.group(name="publication")
def publication_group():
    """Generate publication-ready outputs and academic papers."""
    pass


@publication_group.command("generate")
@click.option("--input", "-i", "input_file", required=True, 
              help="Input data file (JSON or CSV)")
@click.option("--output", "-o", "output_file", required=True,
              help="Output file path")
@click.option("--format", "-f", "output_format", 
              type=click.Choice(["latex", "markdown", "html", "pdf"]),
              default="latex", help="Output format")
@click.option("--template", "-t", "template_type",
              type=click.Choice(["ieee_conference", "acm", "springer_lncs", "elsevier", "generic"]),
              default="ieee_conference", help="LaTeX template type")
@click.option("--title", help="Paper title")
@click.option("--authors", help="Comma-separated list of authors")
@click.option("--abstract", help="Paper abstract")
@click.option("--methodologies", help="Comma-separated list of methodologies used")
def generate_paper(input_file: str, output_file: str, output_format: str,
                  template_type: str, title: Optional[str], authors: Optional[str],
                  abstract: Optional[str], methodologies: Optional[str]):
    """Generate academic paper from analysis results."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Loading data...", total=None)
        
        # Load input data
        try:
            if input_file.endswith('.json'):
                with open(input_file, 'r') as f:
                    data = json.load(f)
            elif input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
                data = {'analysis_results': df.to_dict()}
            else:
                raise ValueError("Unsupported input format. Use JSON or CSV.")
        except Exception as e:
            console.print(f"[red]Error loading input file: {e}[/red]")
            return
        
        progress.update(task, description="Generating paper...")
        
        # Set default values
        if not title:
            title = "ESCAI Framework Analysis Results"
        authors_list: List[str]
        if not authors:
            authors_list = ["Research Team"]
            authors = "Research Team"
        else:
            authors_list = [author.strip() for author in authors.split(",")]
        if not abstract:
            abstract = "This paper presents analysis results from the ESCAI Framework for autonomous agent monitoring and analysis."
        
        # Parse methodologies
        methodology_list = []
        if methodologies:
            methodology_list = [m.strip() for m in methodologies.split(",")]
        
        # Prepare data for formatting
        format_data = {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'analysis_results': data.get('analysis_results', {}),
            'methodologies': methodology_list
        }
        
        try:
            # Generate paper
            if output_format == "pdf":
                # Generate LaTeX first, then compile to PDF
                latex_content = format_for_publication(format_data, "latex", template_type)
                
                # Save LaTeX file
                latex_file = output_file.replace('.pdf', '.tex')
                with open(latex_file, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                
                # Try to compile to PDF
                progress.update(task, description="Compiling PDF...")
                try:
                    import subprocess
                    result = subprocess.run(['pdflatex', latex_file], 
                                          capture_output=True, text=True, cwd=os.path.dirname(latex_file))
                    if result.returncode == 0:
                        console.print(f"[green]✓[/green] PDF generated: {output_file}")
                    else:
                        console.print(f"[yellow]LaTeX file generated: {latex_file}[/yellow]")
                        console.print("[yellow]PDF compilation failed. Please compile manually.[/yellow]")
                except FileNotFoundError:
                    console.print(f"[yellow]LaTeX file generated: {latex_file}[/yellow]")
                    console.print("[yellow]pdflatex not found. Please compile manually.[/yellow]")
            else:
                # Generate other formats
                content = format_for_publication(format_data, output_format, template_type)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                console.print(f"[green]✓[/green] Paper generated: {output_file}")
        
        except Exception as e:
            console.print(f"[red]Error generating paper: {e}[/red]")
            return
    
    # Show summary
    _show_generation_summary(output_file, output_format, template_type)


@publication_group.command("report")
@click.option("--input", "-i", "input_file", required=True,
              help="Input data file (JSON or CSV)")
@click.option("--output", "-o", "output_file", required=True,
              help="Output report file")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["latex", "markdown", "html"]),
              default="latex", help="Output format")
@click.option("--include-stats", is_flag=True, default=True,
              help="Include statistical analysis")
@click.option("--include-plots", is_flag=True, default=True,
              help="Include plots and visualizations")
def generate_report(input_file: str, output_file: str, output_format: str,
                   include_stats: bool, include_plots: bool):
    """Generate comprehensive statistical report."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Loading data...", total=None)
        
        # Load data
        try:
            data_dict = {}
            
            if input_file.endswith('.json'):
                with open(input_file, 'r') as f:
                    raw_data = json.load(f)
                
                # Convert to DataFrames if needed
                for key, value in raw_data.items():
                    if isinstance(value, list) and value:
                        data_dict[key] = pd.DataFrame(value)
                    elif isinstance(value, dict):
                        data_dict[key] = pd.DataFrame([value])
            
            elif input_file.endswith('.csv'):
                data_dict['agent_performance'] = pd.read_csv(input_file)
            
        except Exception as e:
            console.print(f"[red]Error loading data: {e}[/red]")
            return
        
        progress.update(task, description="Generating statistical report...")
        
        # Generate report
        try:
            report_generator = StatisticalReportGenerator()
            
            methodologies = ["statistical_analysis", "epistemic_extraction"]
            if any("causal" in key.lower() for key in data_dict.keys()):
                methodologies.append("causal_analysis")
            
            report_content = report_generator.generate_full_report(
                data_dict, 
                methodologies=methodologies
            )
            
            # Add bibliography
            citations = report_generator.generate_methodology_citations(methodologies)
            bibliography = report_generator.citation_db.generate_bibliography(
                citations, "bibtex" if output_format == "latex" else "apa"
            )
            
            if output_format == "latex":
                # Create complete LaTeX document
                template_manager = LatexTemplateManager()
                full_document = template_manager.generate_document(
                    "generic",
                    "ESCAI Framework Statistical Report",
                    ["ESCAI Research Team"],
                    "Comprehensive statistical analysis of ESCAI monitoring data.",
                    report_content
                )
                
                # Save bibliography separately
                bib_file = output_file.replace('.tex', '.bib')
                with open(bib_file, 'w', encoding='utf-8') as f:
                    f.write(bibliography)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(full_document)
            
            else:
                # For other formats, append bibliography
                full_content = report_content + "\n\n" + bibliography
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(full_content)
            
            console.print(f"[green]✓[/green] Statistical report generated: {output_file}")
            
        except Exception as e:
            console.print(f"[red]Error generating report: {e}[/red]")
            return
    
    _show_report_summary(output_file, len(data_dict))


@publication_group.command("citations")
@click.option("--methodology", "-m", multiple=True,
              help="Methodology to get citations for")
@click.option("--format", "-f", "citation_format",
              type=click.Choice(["bibtex", "apa", "ieee"]),
              default="bibtex", help="Citation format")
@click.option("--output", "-o", "output_file",
              help="Output file for bibliography")
@click.option("--search", "-s", help="Search citations by keyword")
def manage_citations(methodology: List[str], citation_format: str,
                    output_file: Optional[str], search: Optional[str]):
    """Manage citations and generate bibliographies."""
    
    citation_db = CitationDatabase()
    methodology_gen = MethodologyCitationGenerator(citation_db)
    
    if search:
        # Search citations
        results = citation_db.search_citations(search)
        
        if results:
            console.print(f"[green]Found {len(results)} citations:[/green]")
            
            table = Table(title="Citation Search Results")
            table.add_column("Key", style="cyan")
            table.add_column("Title", style="white")
            table.add_column("Authors", style="yellow")
            table.add_column("Year", style="green")
            
            for key in results:
                citation = citation_db.get_citation(key)
                if citation:
                    authors = ", ".join([author.last_name for author in citation.authors[:3]])
                    if len(citation.authors) > 3:
                        authors += " et al."
                    
                    table.add_row(
                        key,
                        citation.title[:50] + "..." if len(citation.title) > 50 else citation.title,
                        authors,
                        str(citation.year) if citation.year else "N/A"
                    )
            
            console.print(table)
        else:
            console.print(f"[yellow]No citations found for '{search}'[/yellow]")
        
        return
    
    # Get citations for methodologies
    all_citations = []
    
    if methodology:
        for method in methodology:
            citations = methodology_gen.get_methodology_citations(method)
            all_citations.extend(citations)
            
            console.print(f"[green]Citations for {method}:[/green]")
            for citation_key in citations:
                console.print(f"  - {citation_key}")
    else:
        # Show all available methodologies
        available_methods = [
            "causal_analysis", "pattern_mining", "epistemic_extraction",
            "multiagent_monitoring", "statistical_analysis", "time_series_analysis",
            "real_time_monitoring", "agent_safety_monitoring", "agent_governance",
            "neurosymbolic_agents", "runtime_monitoring"
        ]
        
        console.print("[green]Available methodologies:[/green]")
        for method in available_methods:
            citations = methodology_gen.get_methodology_citations(method)
            console.print(f"  - {method}: {len(citations)} citations")
        
        return
    
    # Generate bibliography
    if all_citations:
        all_citations = list(set(all_citations))  # Remove duplicates
        
        bibliography = citation_db.generate_bibliography(all_citations, citation_format)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(bibliography)
            console.print(f"[green]✓[/green] Bibliography saved to: {output_file}")
        else:
            console.print("\n[green]Generated Bibliography:[/green]")
            console.print(bibliography)


@publication_group.command("templates")
@click.option("--list", "list_templates", is_flag=True,
              help="List available templates")
@click.option("--show", help="Show template details")
@click.option("--output", "-o", help="Generate sample document")
def manage_templates(list_templates: bool, show: Optional[str], output: Optional[str]):
    """Manage LaTeX templates for academic papers."""
    
    template_manager = LatexTemplateManager()
    
    if list_templates:
        templates = template_manager.list_templates()
        
        table = Table(title="Available LaTeX Templates")
        table.add_column("Template", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Document Class", style="yellow")
        table.add_column("Bibliography Style", style="green")
        
        for template_key in templates:
            template = template_manager.get_template(template_key)
            table.add_row(
                template_key,
                template.name,
                template.document_class.replace("\\documentclass", "").strip("{}[]"),
                template.bibliography_style
            )
        
        console.print(table)
        return
    
    if show:
        try:
            template = template_manager.get_template(show)
            
            panel_content = f"""
**Document Class:** {template.document_class}
**Bibliography Style:** {template.bibliography_style}

**Packages:**
{chr(10).join(template.packages)}

**Content Structure:**
{chr(10).join(template.content_structure)}
"""
            
            console.print(Panel(panel_content, title=f"Template: {template.name}"))
            
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
        return
    
    if output:
        # Generate sample document
        try:
            sample_content = """\\section{Introduction}

This is a sample document generated using the ESCAI Framework publication system.

\\section{Methodology}

The analysis employs standard statistical methods and the ESCAI Framework for agent monitoring.

\\section{Results}

Sample results would be presented here with appropriate tables and figures.

\\section{Conclusion}

This demonstrates the template structure for academic publications."""
            
            document = template_manager.generate_document(
                "ieee_conference",  # Default template
                "Sample ESCAI Framework Paper",
                ["John Doe", "Jane Smith"],
                "This is a sample abstract demonstrating the ESCAI Framework publication system.",
                sample_content,
                ["ESCAI", "agent monitoring", "academic publishing"]
            )
            
            with open(output, 'w', encoding='utf-8') as f:
                f.write(document)
            
            console.print(f"[green]✓[/green] Sample document generated: {output}")
            
        except Exception as e:
            console.print(f"[red]Error generating sample: {e}[/red]")


def _show_generation_summary(output_file: str, output_format: str, template_type: str):
    """Show summary of paper generation."""
    
    summary_data = {
        "Output File": output_file,
        "Format": output_format.upper(),
        "Template": template_type,
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    table = Table(title="Paper Generation Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in summary_data.items():
        table.add_row(key, str(value))
    
    console.print(table)
    
    # Show next steps
    next_steps = Panel(
        """1. Review the generated document
2. Add your specific content and analysis
3. Include figures and tables as needed
4. Compile to PDF if using LaTeX
5. Submit to your target venue""",
        title="Next Steps",
        border_style="green"
    )
    
    console.print(next_steps)


def _show_report_summary(output_file: str, num_datasets: int):
    """Show summary of report generation."""
    
    summary_data = {
        "Report File": output_file,
        "Datasets Analyzed": num_datasets,
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    table = Table(title="Statistical Report Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in summary_data.items():
        table.add_row(key, str(value))
    
    console.print(table)


# Register commands with main CLI
def register_publication_commands(cli_group):
    """Register publication commands with the main CLI."""
    cli_group.add_command(publication_group)