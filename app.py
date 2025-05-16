import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.express as px
from text_extraction import extract_text_from_pdf, extract_text_from_docx
from resume_analyzer import analyze_resume, match_keywords, check_format, detect_resume_sections, parse_job_description
from utils import generate_suggestions, generate_report, check_plagiarism
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==============================================
# Page Configuration
# ==============================================
st.set_page_config(
    page_title="Resume ATS Checker",
    page_icon="📝",
    layout="wide"
)

# ==============================================
# Sidebar Settings
# ==============================================
with st.sidebar:
    st.title("⚙️ Analysis Settings")
    
    # Keyword matching settings
    if 'keyword_weight' not in st.session_state:
        st.session_state.keyword_weight = 1.0
        
    st.session_state.keyword_weight = st.slider(
        "Keyword Matching Weight",
        min_value=0.5,
        max_value=1.5,
        value=st.session_state.keyword_weight,
        step=0.1,
        help="Adjust the importance of keyword matching in the overall score"
    )
    
    # ATS compatibility settings
    if 'ats_weight' not in st.session_state:
        st.session_state.ats_weight = 1.0
        
    st.session_state.ats_weight = st.slider(
        "ATS Compatibility Weight",
        min_value=0.5,
        max_value=1.5,
        value=st.session_state.ats_weight,
        step=0.1,
        help="Adjust the importance of ATS compatibility in the overall score"
    )
    
    # Format score settings
    if 'format_weight' not in st.session_state:
        st.session_state.format_weight = 1.0
        
    st.session_state.format_weight = st.slider(
        "Format & Structure Weight",
        min_value=0.5,
        max_value=1.5,
        value=st.session_state.format_weight,
        step=0.1,
        help="Adjust the importance of formatting and structure in the overall score"
    )
    
    # Scoring factor explanations
    with st.expander("💡 What these factors mean"):
        st.markdown("""
        **Keyword Matching**: How well your resume matches the job description.
        - Higher values emphasize the importance of having specific keywords from the job description.
        
        **ATS Compatibility**: How well your resume is likely to be parsed by ATS systems.
        - Higher values focus on proper formatting, section headers, and content structure.
        
        **Format & Structure**: The readability and organization of your resume.
        - Higher values emphasize bullet points, consistent formatting, and proper layout.
        """)
    
    st.markdown("---")
    with st.expander("Privacy Notice"):
        st.markdown("""
        **Your data is safe:**
        - All processing happens in your browser
        - No resume data is stored or saved
        - Files are deleted after analysis
        - No personal data is collected
        """)
    
    st.markdown("Made with ❤️ using Streamlit")

# ==============================================
# Main App Content
# ==============================================
st.title("📝 Resume ATS Checker and Scorer")
st.markdown("""
This application helps you analyze your resume against Applicant Tracking Systems (ATS).
Upload your resume and a job description to get detailed feedback and improvement suggestions.
""")

# File Upload Columns
col1, col2 = st.columns(2)

# Initialize resume_text
resume_text = None

# Resume Upload Section
with col1:
    st.subheader("Upload Your Resume")
    resume_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 5MB
    
    if resume_file and resume_file.size > MAX_FILE_SIZE:
        st.error("File too large (max 5MB allowed)")
        resume_file = None
    
    if resume_file is not None:
        # Get file details
        file_details = {
            "Filename": resume_file.name, 
            "FileType": resume_file.type, 
            "FileSize": f"{round(resume_file.size / 1024, 2)} KB"
        }
        st.write(file_details)
        
        # Extract text from resume
        try:
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_text_from_docx(resume_file)
            else:
                st.error("Unsupported file format. Please upload a PDF or DOCX file.")
                resume_text = None
                
            if resume_text:
                st.success("Resume text extracted successfully!")
                with st.expander("View extracted text"):
                    st.write(resume_text)
        except Exception as e:
            st.error(f"Error extracting text from resume: {e}")
            resume_text = None

# Job Description Section
with col2:
    st.subheader("Paste Job Description")
    job_description = st.text_area("Enter the job description here", height=250)

# Analysis Button
analyze_button = st.button("Analyze Resume", disabled=(resume_file is None or not job_description))

# ==============================================
# Analysis Results Section
# ==============================================
if analyze_button and resume_text and job_description:
    st.header("Analysis Results")
     # Add the warning message here (right after the header)
    st.warning("""
    **Important Note About Results**:
    This analysis provides approximate guidance, not definitive scoring. Real ATS systems may behave differently because:

    - We use simplified text matching (not semantic analysis)
    - Some formatting may not be perfectly interpreted  
    - Industry-specific terminology may not be recognized
    - Results are meant for improvement suggestions, not absolute evaluation

    Use this as a helpful guide rather than an exact assessment.
    """)
    with st.spinner("Analyzing your resume..."):
        # Perform all analyses
        ats_score, ats_feedback = analyze_resume(resume_text)
        keyword_matches, missing_keywords, keyword_score = match_keywords(resume_text, job_description)
        format_score, format_feedback = check_format(resume_text)
        
        # Apply weights and calculate overall score
        weighted_ats_score = ats_score * st.session_state.ats_weight
        weighted_keyword_score = keyword_score * st.session_state.keyword_weight
        weighted_format_score = format_score * st.session_state.format_weight
        sum_weights = (st.session_state.ats_weight + st.session_state.keyword_weight + st.session_state.format_weight)
        overall_score = (weighted_ats_score + weighted_keyword_score + weighted_format_score) / sum_weights
        
        # Additional analyses
        sections = detect_resume_sections(resume_text)
        jd_requirements = parse_job_description(job_description)
        plagiarism_result = check_plagiarism(resume_text)

        # Display Scores
        score_col1, score_col2, score_col3, score_col4 = st.columns(4)
        with score_col1: st.metric("Overall Score", f"{overall_score:.1f}/10")
        with score_col2: st.metric("ATS Compatibility", f"{ats_score:.1f}/10")
        with score_col3: st.metric("Keyword Match", f"{keyword_score:.1f}/10")
        with score_col4: st.metric("Format & Structure", f"{format_score:.1f}/10")
        
        # ==============================================
        # Visualization Tabs
        # ==============================================
        st.subheader("Detailed Analysis Dashboard")
        tab1, tab2, tab3= st.tabs(["Score Overview", "Keyword Analysis", "Resume Structure"])
        
        # Tab 1: Score Overview
        with tab1:
            # Gauge Chart
            overall_gauge = {
                'data': [{
                    'type': 'indicator',
                    'mode': 'gauge+number+delta',
                    'value': overall_score,
                    'title': {
                                'text': '<b>OVERALL ATS SCORE</b>',
                                'font': {
                                    'size': 28,
                                    'family': "Arial Black",
                                    'color': "navy"
                                }
                                
                            },
                    'gauge': {
                        'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': 'darkblue'},
                        'bar': {'color': 'darkblue'},
                        'bgcolor': 'white',
                        'borderwidth': 2,
                        'bordercolor': 'gray',
                        'steps': [
                            {'range': [0, 3], 'color': 'red'},
                            {'range': [3, 5], 'color': 'orange'},
                            {'range': [5, 7], 'color': 'yellow'},
                            {'range': [7, 8.5], 'color': 'lightgreen'},
                            {'range': [8.5, 10], 'color': 'green'},
                        ],
                        'threshold': {
                            'line': {'color': 'navy', 'width': 4},
                            'thickness': 0.75,
                            'value': 8.5
                        }
                    },
                    'delta': {
                        'reference': 8.5,
                        'increasing': {'color': 'green'},
                        'decreasing': {'color': 'red'}
                    }
                }],
                'layout': {
                    'height': 300,
                    'margin': {'t': 25, 'r': 25, 'l': 25, 'b': 25},
                    'paper_bgcolor': 'white',
                    'font': {'color': 'darkblue', 'family': 'Arial'}
                }
            }
            st.plotly_chart(overall_gauge)
            
            # Bar Chart
            scores_df = pd.DataFrame({
                'Category': ['ATS Compatibility', 'Keyword Match', 'Format & Structure'],
                'Score': [ats_score, keyword_score, format_score],
                'Weight': [st.session_state.ats_weight, st.session_state.keyword_weight, st.session_state.format_weight],
                'Weighted Score': [
                    ats_score * st.session_state.ats_weight / st.session_state.ats_weight,
                    keyword_score * st.session_state.keyword_weight / st.session_state.keyword_weight,
                    format_score * st.session_state.format_weight / st.session_state.format_weight
                ]
            })
            
            colors = []
            for score in scores_df['Score']:
                if score < 5: colors.append('rgba(255, 0, 0, 0.8)')
                elif score < 7: colors.append('rgba(255, 165, 0, 0.8)')
                elif score < 8.5: colors.append('rgba(0, 150, 0, 0.8)')
                else: colors.append('rgba(0, 100, 0, 0.8)')
            
            fig = px.bar(
                scores_df, 
                x='Category', 
                y='Score',
                text='Score',
                title="<b>Detailed Score Analysis</b>",
                labels={'Score': 'Score (0-10)', 'Category': ''},
                range_y=[0, 10],
                color='Category',
                color_discrete_map={
                    'ATS Compatibility': colors[0],
                    'Keyword Match': colors[1],
                    'Format & Structure': colors[2]
                }
            )
            
            fig.update_traces(
                texttemplate='%{y:.1f}',
                textposition='inside',
                textfont=dict(size=14, color='white'),
                marker=dict(line=dict(width=2, color='rgba(0,0,0,0.3)')),
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}/10<extra></extra>'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(240,240,240,0.6)',
                height=400,
                showlegend=False,
                font=dict(family="Arial", size=14),
                yaxis=dict(
                    tickvals=[0, 2, 4, 6, 8, 10],
                    ticktext=['0', '2', '4', '6', '8', '10'],
                    gridcolor='rgba(200,200,200,0.2)',
                    title=dict(font=dict(size=16))
                ),
                xaxis=dict(title=dict(font=dict(size=16))),
                shapes=[dict(
                    type='line',
                    yref='y', y0=7, y1=7,
                    xref='paper', x0=0, x1=1,
                    line=dict(color='rgba(50, 50, 50, 0.4)', width=2, dash='dash')
                )],
                annotations=[dict(
                    x=0.5, y=7.3,
                    xref='paper', yref='y',
                    text='Good Score Threshold',
                    showarrow=False,
                    font=dict(size=12, color='rgba(50, 50, 50, 0.8)')
                )]
            )
            st.plotly_chart(fig)
            
            # Radar Chart
            radar_fig = {
                'data': [{
                    'type': 'scatterpolar',
                    'r': [ats_score, keyword_score, format_score, 
                          min(10, overall_score * 1.05),
                          min(10, (ats_score + keyword_score) / 2)],
                    'theta': ['ATS<br>Compatibility', 'Keyword<br>Match', 'Format &<br>Structure', 
                              'Overall<br>Score', 'Content<br>Quality'],
                    'fill': 'toself',
                    'fillcolor': 'rgba(67, 147, 195, 0.2)',
                    'line': {'color': 'rgba(6, 82, 121, 0.8)'},
                    'name': 'Resume Score'
                }, {
                    'type': 'scatterpolar',
                    'r': [8.5, 8.5, 8.5, 8.5, 8.5],
                    'theta': ['ATS<br>Compatibility', 'Keyword<br>Match', 'Format &<br>Structure', 
                              'Overall<br>Score', 'Content<br>Quality'],
                    'fill': 'toself',
                    'fillcolor': 'rgba(44, 160, 44, 0.1)',
                    'line': {'color': 'rgba(44, 160, 44, 0.5)', 'dash': 'dash'},
                    'name': 'Target Score'
                }]
            }
            
            radar_layout = {
                'title': 'Multi-dimensional Analysis',
                'polar': {'radialaxis': {'visible': True, 'range': [0, 10]}},
                'showlegend': True,
                'legend': {'x': 0.8, 'y': 0.95},
                'height': 500,
                'margin': {'t': 50, 'b': 50}
            }
            st.plotly_chart({'data': radar_fig['data'], 'layout': radar_layout})
        
        # Tab 2: Keyword Analysis
        with tab2:
            st.subheader("Keyword Match Analysis")
            
            # Matched Keywords
            st.write("✅ **Matched Keywords**")
            if keyword_matches:
                cols = st.columns(4)
                for i, keyword in enumerate(keyword_matches[:20]):
                    cols[i%4].success(f"✓ {keyword}")
            else:
                st.write("No matching keywords found.")
            
            # Missing Keywords
            st.subheader("❌ **Missing Keywords**")
            if missing_keywords:
                missing_cols = st.columns(3)
                high_importance = missing_keywords[:min(5, len(missing_keywords))]
                medium_importance = missing_keywords[min(5, len(missing_keywords)):min(10, len(missing_keywords))]
                low_importance = missing_keywords[min(10, len(missing_keywords)):]
                
                with missing_cols[0]:
                    st.markdown("**High Priority**")
                    for keyword in high_importance:
                        st.markdown(f"🔴 {keyword}")
                    if not high_importance:
                        st.markdown("None - Great job!")
                
                with missing_cols[1]:
                    st.markdown("**Medium Priority**")
                    for keyword in medium_importance:
                        st.markdown(f"🟠 {keyword}")
                    if not medium_importance:
                        st.markdown("None - Great job!")
                
                with missing_cols[2]:
                    st.markdown("**Consider Adding**")
                    for keyword in low_importance:
                        st.markdown(f"🟡 {keyword}")
                    if not low_importance:
                        st.markdown("None - Great job!")
                
                st.info("""
                **How to use these keywords:**
                - Add high priority keywords exactly as shown
                - Integrate medium priority keywords where relevant
                - Consider adding lower priority keywords if they reflect your actual experience
                - Don't just list keywords - use them naturally in context
                """)
            else:
                st.success("Great job! Your resume contains all important keywords from the job description.")
        
        # Tab 3: Resume Structure
        with tab3:
            st.subheader("Resume Section Analysis")
            
            col1, col2 = st.columns(2)
            for section, present in sections.items():
                (col1 if present else col2).write(
                    f"{'✅' if present else '❌'} {section.capitalize()} "
                    f"{'found' if present else 'missing'}"
                )
            
            st.subheader("Format & Structure")
            st.write(format_feedback)
            
            st.subheader("Plagiarism Check")
            st.write(plagiarism_result)
        
        
        
        # ==============================================
        # Detailed Feedback and Suggestions
        # ==============================================
        st.header("Detailed Feedback")
        
        with st.expander("ATS Compatibility Analysis"):
            st.write(ats_feedback)
            
        with st.expander("Format & Structure Analysis"):
            st.write(format_feedback)
            
        with st.expander("Plagiarism Check"):
            st.write(plagiarism_result)
        
        st.header("Improvement Suggestions")
        suggestions = generate_suggestions(
            resume_text, 
            job_description, 
            ats_score, 
            keyword_matches, 
            missing_keywords, 
            format_score
        )
        
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"{i}. {suggestion}")
        
        # Report Download
        st.header("Download Report")
        report = generate_report(
            resume_text,
            job_description,
            overall_score,
            ats_score,
            ats_feedback,
            keyword_score,
            keyword_matches,
            missing_keywords,
            format_score,
            format_feedback,
            suggestions
        )
        
        report_bytes = report.encode()
        st.download_button(
            label="Download Analysis Report",
            data=report_bytes,
            file_name="resume_ats_analysis.txt",
            mime="text/plain"
        )

# ==============================================
# Help and Information Sections
# ==============================================
with st.expander("How to use this ATS Resume Checker"):
    st.markdown("""
    ### Instructions
    1. **Upload your resume** in PDF or DOCX format
    2. **Paste the job description** for the position you're applying to
    3. Click **Analyze Resume** to get detailed feedback
    4. Review your scores and suggestions for improvement
    5. Download the analysis report for future reference
    
    ### Why this matters
    Applicant Tracking Systems (ATS) are used by most employers to screen resumes before they reach human recruiters.
    Optimizing your resume for ATS can significantly increase your chances of getting an interview.
    
    ### Key factors analyzed
    - **ATS Compatibility**: How well your resume can be parsed by ATS software
    - **Keyword Match**: How well your resume matches the job description
    - **Format & Structure**: Evaluation of your resume's organization and readability
    """)

with st.expander("💡 Additional ATS Information"):
    st.markdown("""
    ### What are ATS systems?
    Applicant Tracking Systems (ATS) are software applications that employers use to manage their recruitment process. 
    They scan, sort, and rank resumes based on keywords, skills, and other criteria.
    
    ### How our scoring works:
    1. **ATS Compatibility Score**: Evaluates how well your resume can be read by ATS software
    2. **Keyword Match Score**: Compares your resume with the job description
    3. **Format & Structure Score**: Assesses the overall layout and readability
    
    ### Tips for improving your score:
    - Use a clean, simple format with standard section headings
    - Include relevant keywords from the job description
    - Ensure consistent formatting and proper organization
    - Be concise yet comprehensive in describing your experience
    - Use bullet points to highlight achievements
    - Quantify your achievements with numbers where possible
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
