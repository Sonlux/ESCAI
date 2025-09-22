#!/usr/bin/env python3
"""
Example: Generating Academic Publications with Contemporary ESCAI Citations

This example demonstrates how to generate academic papers and reports that include
contemporary research citations relevant to autonomous agent monitoring and analysis.
"""

import json
import tempfile
import os
from pathlib import Path

from escai_framework.cli.utils.publication_formatter import format_for_publication
from escai_framework.cli.utils.citation_manager import CitationDatabase, MethodologyCitationGenerator
from escai_framework.cli.utils.statistical_report_generator import StatisticalReportGenerator


def main():
    """Demonstrate publication generation with contemporary citations."""
    
    print("🔬 ESCAI Framework - Contemporary Citations Example")
    print("=" * 60)
    
    # Sample analysis data
    sample_data = {
        'title': 'Real-time Safety Monitoring of Autonomous Agents using ESCAI Framework',
        'authors': ['Dr. Sarah Chen', 'Prof. Michael Rodriguez', 'Dr. Aisha Patel'],
        'abstract': '''This paper presents a comprehensive approach to real-time safety monitoring 
        of autonomous agents using the ESCAI Framework. We demonstrate novel techniques for 
        out-of-distribution detection in neurosymbolic agents, runtime anomaly detection, 
        and agent governance protocols. Our evaluation across 200 autonomous agents shows 
        significant improvements in safety monitoring accuracy and response time.''',
        'analysis_results': {
            'agent_performance': {
                'total_agents_monitored': 200,
                'safety_incidents_detected': 15,
                'false_positive_rate': 0.023,
                'detection_accuracy': 0.967,
                'average_response_time': 1.2,
                'governance_compliance': 0.994
            },
            'methodology_evaluation': {
                'ood_detection_accuracy': 0.943,
                'runtime_monitoring_overhead': 0.087,
                'agent_governance_effectiveness': 0.991
            }
        },
        'methodologies': [
            'agent_safety_monitoring',
            'neurosymbolic_agents', 
            'agent_governance',
            'runtime_monitoring',
            'statistical_analysis'
        ]
    }
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Working directory: {temp_dir}")
        
        # 1. Generate IEEE Conference Paper
        print("\n1️⃣ Generating IEEE Conference Paper...")
        ieee_paper = os.path.join(temp_dir, "escai_safety_monitoring_ieee.tex")
        
        latex_content = format_for_publication(
            sample_data, 
            "latex", 
            "ieee_conference"
        )
        
        with open(ieee_paper, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"   ✅ IEEE paper generated: {ieee_paper}")
        
        # 2. Generate Bibliography with Contemporary Citations
        print("\n2️⃣ Generating Bibliography with Contemporary Citations...")
        citation_db = CitationDatabase()
        methodology_gen = MethodologyCitationGenerator(citation_db)
        
        # Get all citations for the methodologies used
        all_citations = []
        for methodology in sample_data['methodologies']:
            citations = methodology_gen.get_methodology_citations(methodology)
            all_citations.extend(citations)
        
        all_citations = list(set(all_citations))  # Remove duplicates
        
        # Generate BibTeX bibliography
        bib_file = os.path.join(temp_dir, "references.bib")
        bibliography = citation_db.generate_bibliography(all_citations, "bibtex")
        
        with open(bib_file, 'w', encoding='utf-8') as f:
            f.write(bibliography)
        
        print(f"   ✅ Bibliography generated: {bib_file}")
        print(f"   📚 Total citations: {len(all_citations)}")
        
        # Show contemporary citations included
        contemporary_citations = [
            'ood_detection_2024', 'smartla_2025', 'agent_intelligence_protocol_2025',
            'industry_anomalies_2024', 'agent_simulator_fbdl_2024',
            'causal_inference_cnn_lstm_2025', 'dynamic_architectures_ai_agents_2025'
        ]
        
        included_contemporary = [cite for cite in contemporary_citations if cite in all_citations]
        
        print(f"   🆕 Contemporary citations included: {len(included_contemporary)}")
        for cite in included_contemporary:
            citation_obj = citation_db.get_citation(cite)
            if citation_obj:
                print(f"      • {citation_obj.title[:60]}... ({citation_obj.year})")
        
        # 3. Generate Statistical Report
        print("\n3️⃣ Generating Statistical Report...")
        report_file = os.path.join(temp_dir, "statistical_report.tex")
        
        # Create sample DataFrame for statistical analysis
        import pandas as pd
        sample_df = pd.DataFrame({
            'agent_id': range(1, 201),
            'agent_type': (['neurosymbolic'] * 80 + ['symbolic'] * 60 + ['neural'] * 60),
            'safety_score': [0.9 + 0.1 * ((i * 7) % 11) / 10 for i in range(200)],
            'detection_accuracy': [0.85 + 0.15 * ((i * 3) % 7) / 6 for i in range(200)],
            'response_time': [0.5 + 2.0 * ((i * 5) % 13) / 12 for i in range(200)],
            'governance_compliance': [0.95 + 0.05 * ((i * 11) % 17) / 16 for i in range(200)]
        })
        
        data_dict = {'agent_performance': sample_df}
        
        report_generator = StatisticalReportGenerator()
        report_content = report_generator.generate_full_report(
            data_dict,
            title="ESCAI Framework Safety Monitoring Statistical Analysis",
            methodologies=sample_data['methodologies']
        )
        
        # Create complete LaTeX document for report
        from escai_framework.cli.utils.latex_templates import LatexTemplateManager
        template_manager = LatexTemplateManager()
        
        full_report = template_manager.generate_document(
            "generic",
            "ESCAI Framework Safety Monitoring Statistical Analysis",
            sample_data['authors'],
            "Statistical analysis of safety monitoring performance across 200 autonomous agents.",
            report_content,
            ["ESCAI", "safety monitoring", "autonomous agents", "statistical analysis"]
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        print(f"   ✅ Statistical report generated: {report_file}")
        
        # 4. Generate Markdown Summary
        print("\n4️⃣ Generating Markdown Summary...")
        markdown_file = os.path.join(temp_dir, "publication_summary.md")
        
        markdown_content = format_for_publication(sample_data, "markdown")
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"   ✅ Markdown summary generated: {markdown_file}")
        
        # 5. Show methodology coverage
        print("\n5️⃣ Methodology Coverage Analysis:")
        for methodology in sample_data['methodologies']:
            citations = methodology_gen.get_methodology_citations(methodology)
            print(f"   📊 {methodology}: {len(citations)} citations")
            
            # Show contemporary citations for this methodology
            contemporary_in_method = [cite for cite in citations if cite in contemporary_citations]
            if contemporary_in_method:
                print(f"      🆕 Contemporary: {', '.join(contemporary_in_method)}")
        
        print(f"\n📋 Summary:")
        print(f"   • IEEE Conference Paper: {os.path.basename(ieee_paper)}")
        print(f"   • Bibliography: {os.path.basename(bib_file)}")
        print(f"   • Statistical Report: {os.path.basename(report_file)}")
        print(f"   • Markdown Summary: {os.path.basename(markdown_file)}")
        print(f"   • Total Contemporary Citations: {len(included_contemporary)}")
        
        # Show file sizes
        print(f"\n📏 File Sizes:")
        for file_path in [ieee_paper, bib_file, report_file, markdown_file]:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"   • {os.path.basename(file_path)}: {size:,} bytes")
        
        print(f"\n✨ All files generated successfully in: {temp_dir}")
        print("   Note: Files are in a temporary directory and will be cleaned up automatically.")
        print("   In practice, specify a permanent directory for your publications.")


if __name__ == "__main__":
    main()