import datetime
import re
from difflib import SequenceMatcher
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_lg")
def generate_suggestions(resume_text, job_description, ats_score, matched_keywords, missing_keywords, format_score):
    """
    Generate improvement suggestions based on analysis results
    
    Args:
        resume_text: Text content of the resume
        job_description: Text content of the job description
        ats_score: ATS compatibility score
        matched_keywords: List of matched keywords
        missing_keywords: List of missing keywords
        format_score: Format and structure score
        
    Returns:
        list: List of improvement suggestions
    """
    suggestions = []
     # SpaCy-based content analysis
    doc = nlp(resume_text[:3000])  # Process first 3000 chars
    
    # Action verb check
    action_verbs = {'develop', 'implement', 'create', 'lead', 'manage'}
    found_verbs = sum(1 for token in doc if token.lemma_ in action_verbs)
    if found_verbs < 5:
        suggestions.append(f"⚠️ Only {found_verbs} strong action verbs found. Use more verbs like 'developed', 'implemented', 'led'.")
    
    
      # NEW: Entity-based suggestions
    entities = {ent.label_: [] for ent in doc.ents}
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    
    if not entities.get("ORG"):
        suggestions.append("🏢 Add more company/organization names in your experience section.")
    if not entities.get("DATE"):
        suggestions.append("📅 Include more dates in your work history.")
    
    # Keyword-related suggestions
    if missing_keywords:
        suggestions.append(f"Add these missing keywords from the job description: {', '.join(missing_keywords[:5])}.")
        
        if len(missing_keywords) > 5:
            suggestions.append(f"Consider incorporating more job-specific terms like: {', '.join(missing_keywords[5:10])}.")
     # NEW: Check for missing sections
    from resume_analyzer import detect_resume_sections
    sections = detect_resume_sections(resume_text)
    
    if not sections['contact']:
        suggestions.append("❌ Missing clear contact information section. Add your email and phone number at the top.")
    if not sections['summary']:
        suggestions.append("ℹ️ Consider adding a professional summary section to highlight your key qualifications.")
    if not sections['experience']:
        suggestions.append("❌ Missing work experience section. Include your professional history.")
    if not sections['education']:
        suggestions.append("❌ Missing education section. Include your academic background.")
    if not sections['skills']:
        suggestions.append("ℹ️ Consider adding a dedicated skills section to showcase your competencies.")
    
    # NEW: Plagiarism check
    plagiarism_result = check_plagiarism(resume_text)
    if "similarity" in plagiarism_result:
        suggestions.append(plagiarism_result)
    
    # ATS-specific suggestions
    if ats_score < 7:
        suggestions.append("Use a simpler resume format that is more ATS-friendly. Avoid tables, columns, headers/footers, and complex formatting.")
        suggestions.append("Make sure your contact information is clearly listed at the top of your resume.")
        suggestions.append("Use standard section headings (e.g., 'Experience', 'Education', 'Skills') that ATS systems can easily recognize.")
    
    # Format-related suggestions
    if format_score < 7:
        suggestions.append("Use bullet points to describe your achievements and responsibilities for better readability.")
        suggestions.append("Keep your resume concise, ideally 1-2 pages depending on your experience level.")
        suggestions.append("Ensure consistent formatting for dates, bullet points, and section headers.")
    
    # Content-related suggestions
    if len(resume_text.split()) < 300:
        suggestions.append("Your resume content seems light. Add more relevant experiences, skills, and accomplishments.")
    elif len(resume_text.split()) > 700:
        suggestions.append("Your resume may be too detailed. Focus on the most relevant information for this specific job.")
    
    # Check for quantifiable achievements
    number_pattern = r'\b\d+%|\$\d+|\d+ years|\d+ months|\d+ weeks|\d+ days|\d+ hours|\d+ projects\b'
    quantifiable = len(re.findall(number_pattern, resume_text, re.IGNORECASE))
    
    if quantifiable < 3:
        suggestions.append("Add more quantifiable achievements (numbers, percentages, timeframes) to strengthen your impact.")
    

     # NEW: Job description parsing suggestions
    from resume_analyzer import parse_job_description
    jd_requirements = parse_job_description(job_description)
    
    if jd_requirements['skills'] and not any(skill in ' '.join(matched_keywords) for skill in jd_requirements['skills']):
        suggestions.append(f"Important skills mentioned in job description: {', '.join(jd_requirements['skills'][:5])}") 


    # General suggestions
    suggestions.append("Tailor your resume for each job application rather than using a generic version.")
    suggestions.append("Consider adding a brief, targeted summary section at the top of your resume.")
    
    if len(matched_keywords) / (len(matched_keywords) + len(missing_keywords)) < 0.6:
        suggestions.append("Your resume and the job description have low keyword similarity. Consider rewriting to better align with the job requirements.")
    
    return suggestions

# NEW FUNCTION: Plagiarism check
def check_plagiarism(text):
    """Check for similarity with common resume templates"""
    common_templates = [
        "Results-driven professional with X years of experience",
        "Detail-oriented individual seeking position",
        "Motivated professional with excellent communication skills",
        "Dynamic team player with strong work ethic",
        "Proven track record of success in"
    ]
    
    max_similarity = 0
    for template in common_templates:
        similarity = SequenceMatcher(None, text[:500], template).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
    
    if max_similarity > 0.7:
        return f"⚠️ High similarity ({max_similarity*100:.1f}%) to common templates detected - personalize your content"
    elif max_similarity > 0.5:
        return f"ℹ️ Moderate similarity ({max_similarity*100:.1f}%) to common templates - consider more original phrasing"
    return "✅ No significant template similarity detected"

def generate_report(resume_text, job_description, overall_score, ats_score, ats_feedback, 
                   keyword_score, matched_keywords, missing_keywords, format_score, format_feedback, suggestions):
    """
    Generate a downloadable report with analysis results
    
    Args:
        All analysis results and scores
        
    Returns:
        str: Formatted report text
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
     # NEW: Parse job description
    from resume_analyzer import parse_job_description, detect_resume_sections
    jd_requirements = parse_job_description(job_description)
    sections = detect_resume_sections(resume_text)
    report = f"""
=======================================================
           RESUME ATS ANALYSIS REPORT
                  {now}
=======================================================

OVERALL SCORE: {overall_score:.1f}/10

-------------------------------------------------------
JOB DESCRIPTION ANALYSIS
-------------------------------------------------------
Key Skills Sought: {', '.join(jd_requirements['skills'][:10]) if jd_requirements['skills'] else 'Not specified'}
Education Requirements: {', '.join(jd_requirements['education']) if jd_requirements['education'] else 'Not specified'}
Experience Requirements: {', '.join(jd_requirements['experience']) if jd_requirements['experience'] else 'Not specified'}

-------------------------------------------------------
RESUME SECTION ANALYSIS
-------------------------------------------------------
Contact Info: {'✅ Present' if sections['contact'] else '❌ Missing'}
Summary: {'✅ Present' if sections['summary'] else '⚠️ Consider adding'}
Experience: {'✅ Present' if sections['experience'] else '❌ Missing'}
Education: {'✅ Present' if sections['education'] else '❌ Missing'}
Skills: {'✅ Present' if sections['skills'] else '⚠️ Consider adding'}
Achievements: {'✅ Present' if sections['achievements'] else '⚠️ Consider adding'}

-------------------------------------------------------
ATS COMPATIBILITY SCORE: {ats_score:.1f}/10
-------------------------------------------------------
{ats_feedback}

-------------------------------------------------------
KEYWORD MATCH SCORE: {keyword_score:.1f}/10
-------------------------------------------------------
MATCHED KEYWORDS:
{", ".join(matched_keywords) if matched_keywords else "None"}

MISSING KEYWORDS:
{", ".join(missing_keywords) if missing_keywords else "None"}

-------------------------------------------------------
FORMAT & STRUCTURE SCORE: {format_score:.1f}/10
-------------------------------------------------------
{format_feedback}

=======================================================
IMPROVEMENT SUGGESTIONS:
=======================================================
"""

    for i, suggestion in enumerate(suggestions, 1):
        report += f"{i}. {suggestion}\n"
    
    report += """
=======================================================
                 NEXT STEPS
=======================================================
1. Implement the suggestions above to improve your resume
2. Re-upload your revised resume for a new analysis
3. Continue refining until you achieve a score of 8 or higher
4. Submit your optimized resume with confidence!

This report was generated by the Resume ATS Checker.
"""

    return report




