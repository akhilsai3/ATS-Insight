import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import streamlit as st
# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

# NLTK data download (keep existing)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')





def analyze_resume(resume_text):
    """
    Analyze resume for ATS compatibility
    
    Args:
        resume_text: Text content of the resume
        
    Returns:
        tuple: (score, feedback) where score is float and feedback is str
    """
    score = 0.0
    feedback = []
    
    # Check if resume has enough content
    if len(resume_text) < 200:
        feedback.append("❌ Resume content seems too short or couldn't be properly extracted.")
        score -= 2
    else:
        feedback.append("✅ Resume has adequate content length.")
        score += 1
    
    # Check for contact information
    contact_pattern = r'(\b[\w.+-]+@[\w-]+\.[\w.-]+\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)'
    contact_info = re.findall(contact_pattern, resume_text)
    
    if contact_info:
        feedback.append("✅ Contact information detected.")
        score += 1
    else:
        feedback.append("❌ No clear contact information found. Make sure your email and phone number are clearly visible.")
        score -= 1
    
    # Check for name (usually at the top)
    # Look for capitalized words at the beginning of resume or beginning of lines
    potential_names = re.findall(r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', resume_text[:500])
    if potential_names:
        feedback.append("✅ Name detected at the beginning of your resume.")
        score += 1
    else:
        feedback.append("❌ Name not clearly identified. Make sure your full name is prominently displayed at the top.")
        score -= 1
    
    # Check for LinkedIn or other social/portfolio links
    link_pattern = r'(linkedin\.com|github\.com|portfolio|website|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
    links = re.findall(link_pattern, resume_text.lower())
    
    if links:
        feedback.append("✅ Professional links detected (LinkedIn, GitHub, portfolio, etc.).")
        score += 1
    else:
        feedback.append("❌ No professional online presence links found. Consider adding LinkedIn/GitHub/portfolio URLs.")
        score -= 0.5
    
    # Check for education section
    education_keywords = ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'phd', 'diploma']
    has_education = any(keyword in resume_text.lower() for keyword in education_keywords)
    
    if has_education:
        feedback.append("✅ Education section detected.")
        score += 1
    else:
        feedback.append("❌ No clear education section found. Include your educational background.")
        score -= 1
    
    # Check for experience section
    experience_keywords = ['experience', 'work', 'job', 'employment', 'career', 'position']
    has_experience = any(keyword in resume_text.lower() for keyword in experience_keywords)
    
    if has_experience:
        feedback.append("✅ Work experience section detected.")
        score += 1
    else:
        feedback.append("❌ No clear work experience section found. Detail your professional background.")
        score -= 1
    
    # Check for skills section
    skills_keywords = ['skills', 'abilities', 'competencies', 'proficiencies', 'technical skills', 'soft skills']
    has_skills = any(keyword in resume_text.lower() for keyword in skills_keywords)
    
    if has_skills:
        feedback.append("✅ Skills section detected.")
        score += 1
    else:
        feedback.append("❌ No clear skills section found. List your relevant skills.")
        score -= 1
    
    # Check for achievements/accomplishments
    achievement_keywords = ['achievement', 'accomplish', 'award', 'honor', 'recognition', 'certification', 'certified', 'improved', 'increased', 'decreased', 'reduced', 'saved', 'led', 'managed', 'delivered']
    achievement_count = sum(1 for keyword in achievement_keywords if keyword in resume_text.lower())
    
    if achievement_count >= 3:
        feedback.append("✅ Good use of achievement-oriented language.")
        score += 1
    elif achievement_count >= 1:
        feedback.append("⚠️ Limited achievement-oriented language. Try to highlight more specific accomplishments.")
        score += 0.5
    else:
        feedback.append("❌ No achievement-oriented language found. Focus on specific accomplishments, not just responsibilities.")
        score -= 1
    
    # Check for action verbs
    action_verbs = ['developed', 'implemented', 'created', 'designed', 'managed', 'led', 'coordinated', 'analyzed', 'established', 'executed', 'initiated', 'generated', 'organized']
    action_verb_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
    
    if action_verb_count >= 5:
        feedback.append("✅ Good use of strong action verbs.")
        score += 1
    elif action_verb_count >= 2:
        feedback.append("⚠️ Limited use of action verbs. Use more powerful verbs to describe your experience.")
        score += 0.5
    else:
        feedback.append("❌ Few or no strong action verbs found. Use powerful verbs to begin bullet points.")
        score -= 1
    
    # Check for excessive use of complex formatting or graphics (inferred from text extraction issues)
    unusual_char_ratio = len(re.findall(r'[^\w\s.,;:?!()-]', resume_text)) / (len(resume_text) + 1)
    
    if unusual_char_ratio > 0.05:
        feedback.append("❌ Possibly excessive use of special characters or formatting that may confuse ATS.")
        score -= 1
    else:
        feedback.append("✅ Text formatting appears to be ATS-friendly.")
        score += 1
    
    # Check for dates in work experience
    date_pattern = r'\b(19|20)\d{2}\b'
    dates = re.findall(date_pattern, resume_text)
    
    if len(dates) >= 2:
        feedback.append("✅ Date information detected in resume.")
        score += 1
    else:
        feedback.append("❌ Limited or no date information found. Include dates for your work experience and education.")
        score -= 1
    
    # Check for file name mentions in the text (potential issue with the PDF/DOCX handling)
    filename_pattern = r'\.(pdf|docx?|txt|rtf)'
    has_filename = re.search(filename_pattern, resume_text, re.IGNORECASE)
    
    if has_filename:
        feedback.append("❌ Possible file name or extension found in content. Check for unintended text extraction artifacts.")
        score -= 1
    
    # Check resume length in words
    word_count = len(re.findall(r'\b\w+\b', resume_text))
    if 300 <= word_count <= 700:
        feedback.append("✅ Resume has an appropriate word count (300-700 words).")
        score += 1
    elif word_count < 300:
        feedback.append("❌ Resume is too short (less than 300 words). Add more relevant details.")
        score -= 1
    else:
        feedback.append("⚠️ Resume may be too long (over 700 words). Consider focusing on most relevant information.")
        score -= 0.5
    
    # Make scoring more stringent - normalize the score to be between 0 and 10
    # Previous formula: normalized_score = min(max((score + 7) / 1.4, 0), 10)
    # New formula with stricter scoring:
    normalized_score = min(max((score + 3) / 1.5, 0), 10)
    
    # Add overall quality assessment with clear recommendations
    if normalized_score >= 8.5:
        feedback.append("\n✨ **EXCELLENT**: Your resume is highly optimized for ATS systems.")
    elif normalized_score >= 7:
        feedback.append("\n✅ **GOOD**: Your resume is reasonably optimized but could use improvements.")
    elif normalized_score >= 5:
        feedback.append("\n⚠️ **NEEDS IMPROVEMENT**: Your resume requires significant optimization for ATS systems.")
    else:
        feedback.append("\n❌ **POOR**: Your resume is not well-optimized for ATS systems. Consider a complete revision.")
    
    return normalized_score, "\n".join(feedback)
# NEW: SpaCy-based section detection
def detect_resume_sections(resume_text):
    """More accurate section detection using flexible header matching"""
    sections = {
        'contact': False,
        'summary': False,
        'experience': False,
        'education': False,
        'skills': False,
        'achievements': False
    }
    
    # Normalize line endings and clean the text
    clean_text = resume_text.replace('\r', '\n').replace('\t', ' ')
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # More flexible contact detection
        if i < 10:  # Check first 10 lines more thoroughly
            if (not sections['contact'] and 
                (re.search(r'\b(email|phone|contact)\b', line_lower) or 
                re.search(r'@\w+\.\w+', line) or 
                re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', line))):
                sections['contact'] = True
        
        # More flexible section header matching
        def is_section_header(text, patterns):
            """Check if text matches any of the header patterns"""
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
            return any(
                re.search(rf'\b{pattern}\b', text.lower()) 
                for pattern in patterns
            )
        
        # Section checks with more pattern variations
        if not sections['summary'] and is_section_header(line, [
            'summary', 'objective', 'profile', 'about me', 'professional summary'
        ]):
            sections['summary'] = True
            
        if not sections['experience'] and is_section_header(line, [
            'experience', 'work history', 'employment', 'professional experience',
            'work experience', 'career history'
        ]):
            sections['experience'] = True
            
        if not sections['education'] and is_section_header(line, [
            'education', 'academic background', 'qualifications', 'degrees',
            'academic qualifications', 'educational background'
        ]):
            sections['education'] = True
            
        if not sections['skills'] and is_section_header(line, [
            'skills', 'technical skills', 'competencies', 'proficiencies',
            'key skills', 'areas of expertise', 'core competencies'
        ]):
            sections['skills'] = True
            
        if not sections['achievements'] and is_section_header(line, [
            'achievements', 'awards', 'honors', 'certifications',
            'accomplishments', 'recognitions', 'publications'
        ]):
            sections['achievements'] = True
    
    # Content-based fallback checks
    content = clean_text.lower()
    if not sections['experience'] and any(
        word in content for word in ['worked', 'job', 'position', 'employed']
    ):
        sections['experience'] = True
        
    if not sections['education'] and any(
        word in content for word in ['university', 'college', 'degree', 'graduated']
    ):
        sections['education'] = True
        
    if not sections['skills'] and any(
        word in content for word in ['proficient', 'skilled', 'expert', 'knowledge']
    ):
        sections['skills'] = True
    
    return sections
def preprocess_text(text):
    """
    Preprocess text for keyword analysis
    
    Args:
        text: Raw text to process
        
    Returns:
        list: Processed tokens
    """
    try:
        # Try standard tokenization first
        tokens = word_tokenize(text.lower())
    except LookupError:
        # Fallback to simple tokenization if NLTK tokenizer fails
        tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback common English stopwords if NLTK stopwords not available
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                     'when', 'where', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'i', 'my', 'me', 'we', 'our',
                     'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'their',
                     'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                     'been', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'to', 'from'}
    
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatize if available
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        # Skip lemmatization if WordNet not available
        pass
    
    return tokens
## NEW: SpaCy-enhanced keyword matcher
def match_keywords(resume_text, job_description):
    """Advanced keyword matching using spaCy's phrase matching"""
    # Create keyword patterns from job description
    doc = nlp(job_description.lower())
    keywords = set()
    
    # Extract noun phrases and important entities
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:  # Only consider short phrases
            keywords.add(chunk.text.lower())
    
    # Add named entities
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECH", "SKILL"]:
            keywords.add(ent.text.lower())
    
    # Process resume
    resume_doc = nlp(resume_text.lower())
    matched = set()
    
    # Find matches using spaCy's similarity
    for keyword in keywords:
        keyword_doc = nlp(keyword)
        for token in resume_doc:
            if token.similarity(keyword_doc) > 0.85:
                matched.add(keyword)
                break
    
    missing = keywords - matched
    score = (len(matched) / len(keywords)) * 10 if keywords else 0
    
    return list(matched), list(missing), score

# NEW: Enhanced job description parser with spaCy
def parse_job_description(job_desc):
    """Extract requirements using spaCy's pattern matching"""
    doc = nlp(job_desc[:3000])  # Process first 3000 chars for efficiency
    
    requirements = {
        'skills': set(),
        'education': set(),
        'experience': set()
    }
    
    # Initialize matcher
    matcher = spacy.matcher.Matcher(nlp.vocab)
    
    # Skill Patterns - CORRECT STRUCTURE
    skill_patterns = [
        [{"LOWER": {"IN": ["skill", "skills", "ability", "abilities"]}}],
        [{"LOWER": "proficient"}, {"LOWER": "in"}],
        [{"LOWER": "experience"}, {"LOWER": "with"}],
        [{"LOWER": "knowledge"}, {"LOWER": "of"}]
    ]
    
    # Education Patterns - CORRECT STRUCTURE
    edu_patterns = [
        [{"LOWER": "degree"}],
        [{"LOWER": "education"}],
        [{"LOWER": "bachelor"}, {"LOWER": {"IN": ["arts", "science"]}}],
        [{"LOWER": "master"}, {"LOWER": {"IN": ["science", "arts"]}}]
    ]
    
    # Experience Patterns - CORRECT STRUCTURE
    exp_patterns = [
        [{"LOWER": "experience"}],
        [{"LOWER": "minimum"}, {"LIKE_NUM": True}, {"LOWER": "years"}],
        [{"LOWER": "at"}, {"LOWER": "least"}, {"LIKE_NUM": True}, {"LOWER": "years"}]
    ]
    
    # Add patterns to matcher - CORRECT USAGE
    matcher.add("SKILL", skill_patterns)
    matcher.add("EDU", edu_patterns)
    matcher.add("EXP", exp_patterns)
    
    # Extract matches
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end].text
        label = nlp.vocab.strings[match_id]
        
        if label == "SKILL":
            requirements['skills'].add(span)
        elif label == "EDU":
            requirements['education'].add(span)
        elif label == "EXP":
            requirements['experience'].add(span)
    
    # Clean results
    def clean_phrases(phrases):
        return [p.strip().title() for p in phrases if len(p.split()) <= 5][:15]
    
    return {k: clean_phrases(v) for k, v in requirements.items()}
# NEW HELPER FUNCTION: Extract keywords
def extract_keywords(text):
    """Helper to extract important keywords from text"""
    tokens = [token.lower() for token in word_tokenize(text) 
             if token.lower() not in stopwords.words('english') and len(token) > 3]
    
    # Get noun phrases and adjectives
    tagged = nltk.pos_tag(tokens)
    keywords = [word for word, pos in tagged 
               if pos.startswith('NN') or pos.startswith('JJ')]
    
    return list(set(keywords))[:15]  # Return top 15 unique keywords

def check_format(resume_text):
    """
    Check resume format and structure
    
    Args:
        resume_text: Text content of the resume
        
    Returns:
        tuple: (score, feedback)
    """
    feedback = []
    score = 0.0
    
    # Check length of resume
    word_count = len(resume_text.split())
    
    if 300 <= word_count <= 700:
        feedback.append("✅ Resume length is appropriate (between 300-700 words).")
        score += 2
    elif word_count < 300:
        feedback.append("❌ Resume may be too short. Consider adding more relevant content.")
        score -= 1
    else:
        feedback.append("❌ Resume may be too long. Consider condensing to focus on most relevant information.")
        score -= 1
    
    # Check for bullet points
    bullet_patterns = [r'•', r'\*', r'-', r'\d+\.']
    bullet_matches = [re.findall(pattern, resume_text) for pattern in bullet_patterns]
    bullet_count = sum(len(matches) for matches in bullet_matches)
    
    if bullet_count >= 10:
        feedback.append("✅ Good use of bullet points, which improve readability.")
        score += 1.5
    elif bullet_count >= 5:
        feedback.append("✅ Bullet points detected, which improve readability.")
        score += 1
    elif bullet_count > 0:
        feedback.append("⚠️ Limited use of bullet points. Consider adding more for better readability.")
        score += 0.5
    else:
        feedback.append("❌ No bullet points detected. Use them to improve readability and ATS parsing.")
        score -= 1
    
    # Check for sections with headers
    lines = resume_text.split('\n')
    potential_headers = [line.strip() for line in lines if line.strip() and len(line.strip()) < 30]
    
    standard_section_headers = [
        'summary', 'professional summary', 'objective', 'experience', 'work experience', 
        'employment history', 'education', 'skills', 'technical skills', 'certifications',
        'projects', 'achievements', 'awards', 'publications', 'languages', 'interests',
        'professional experience', 'qualifications', 'core competencies', 'employment'
    ]
    
    # Count how many standard headers are present
    standard_headers_present = sum(1 for header in potential_headers 
                                   if any(std_header in header.lower() for std_header in standard_section_headers))
    
    if standard_headers_present >= 4:
        feedback.append("✅ Excellent use of standard section headers, easily recognizable by ATS.")
        score += 2
    elif standard_headers_present >= 3:
        feedback.append("✅ Multiple standard section headers detected.")
        score += 1
    elif len(potential_headers) >= 4:
        feedback.append("⚠️ Multiple sections detected, but consider using more standard headers (like 'Experience', 'Education', 'Skills').")
        score += 0.5
    else:
        feedback.append("❌ Few section headers detected. Use clear, standard section headers that ATS systems can recognize.")
        score -= 1
    
    # Check for paragraph length (prefer shorter paragraphs)
    paragraphs = [p for p in resume_text.split('\n\n') if p.strip()]
    long_paragraphs = sum(1 for p in paragraphs if len(p.split()) > 40)
    
    if long_paragraphs == 0:
        feedback.append("✅ Good use of concise paragraphs.")
        score += 1
    elif long_paragraphs == 1:
        feedback.append("⚠️ One long paragraph detected. Consider breaking it into smaller, more readable sections.")
        score -= 0.5
    else:
        feedback.append("❌ Multiple long paragraphs detected. Break content into smaller, scannable sections.")
        score -= 1
    
    # Check for consistent capitalization in headers
    capitalization_styles = set()
    for header in potential_headers:
        if header.isupper():
            capitalization_styles.add("uppercase")
        elif header.islower():
            capitalization_styles.add("lowercase")
        elif header.istitle():
            capitalization_styles.add("title case")
        elif header[0].isupper():
            capitalization_styles.add("sentence case")
    
    if len(capitalization_styles) <= 1:
        feedback.append("✅ Consistent capitalization style in headers.")
        score += 1
    else:
        feedback.append("❌ Inconsistent capitalization in headers. Standardize your capitalization style.")
        score -= 1
    
    # Check for consistent formatting (looking for patterns in the text)
    formatting_consistency = 5  # Default middle score
    
    # Check for consistent date formats
    try:
        date_formats = re.findall(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\.].*?\d{4}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{2}[-/]\d{2}[-/]\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', resume_text)
        
        if date_formats and len(date_formats) >= 2:
            # Check if all dates follow the same format
            first_format_match = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|[-/]', date_formats[0])
            if first_format_match:
                first_format = first_format_match.group()
                
                # Safely check if all dates have the same pattern
                consistent_dates = True
                for date in date_formats:
                    match = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|[-/]', date)
                    if not match or match.group() != first_format:
                        consistent_dates = False
                        break
                
                if consistent_dates:
                    feedback.append("✅ Date formatting is consistent.")
                    formatting_consistency += 1
                else:
                    feedback.append("❌ Inconsistent date formatting. Standardize date formats.")
                    formatting_consistency -= 1
            else:
                feedback.append("❓ Date format analysis inconclusive.")
        else:
            feedback.append("❓ Not enough dates to analyze format consistency.")
    except Exception as e:
        # If date analysis fails, don't penalize the score
        feedback.append("❓ Date format analysis skipped.")
        st.error(f"Error analyzing date formats: {str(e)}")
    
    # Check for consistent punctuation in bullet lists
    bullet_lines = [line.strip() for line in lines if any(re.match(r'\s*' + pattern + r'\s+', line) for pattern in bullet_patterns)]
    
    if bullet_lines:
        # Check if bullet points consistently end with periods or consistently don't
        ends_with_period = [line.endswith('.') for line in bullet_lines]
        if all(ends_with_period) or not any(ends_with_period):
            feedback.append("✅ Consistent punctuation in bullet points.")
            formatting_consistency += 1
        else:
            feedback.append("❌ Inconsistent punctuation in bullet points. Standardize ending punctuation.")
            formatting_consistency -= 1
    
    # Add formatting consistency to score
    score += formatting_consistency
    
    # Check for spacing and organization issues
    paragraph_count = len([p for p in resume_text.split('\n\n') if len(p.strip()) > 0])
    if paragraph_count < 5:
        feedback.append("❌ Not enough distinct sections or paragraphs. Use proper spacing to organize content.")
        score -= 1
    
    # Check for inconsistent spacing between lines
    line_spacings = []
    for i in range(1, len(lines)-1):
        if not lines[i].strip() and not lines[i-1].strip() and not lines[i+1].strip():
            line_spacings.append("triple")
        elif not lines[i].strip() and (not lines[i-1].strip() or not lines[i+1].strip()):
            line_spacings.append("double")
    
    if len(set(line_spacings)) > 1:
        feedback.append("❌ Inconsistent spacing between sections. Standardize your spacing.")
        score -= 1
    
    # Check if the resume has any indication of tables, which are hard for ATS
    table_indicators = ['|', '+---', '----+', '+===', '====+', '+---|', '|---+']
    has_table = any(indicator in resume_text for indicator in table_indicators)
    if has_table:
        feedback.append("❌ Possible table structure detected. Tables are difficult for ATS systems to parse.")
        score -= 2
    
    # Normalize the score to be between 0 and 10 with stricter criteria
    # Original formula: normalized_score = min(max(score / 10 * 10, 0), 10)
    # New stricter formula:
    raw_score = score / 13  # Divide by a higher number to make the scale stricter
    
    # Apply a curve similar to academic grading (harder to get an A)
    if raw_score > 0.9:  # A
        normalized_score = 9 + (raw_score - 0.9) * 10  # 9-10
    elif raw_score > 0.8:  # B
        normalized_score = 8 + (raw_score - 0.8) * 10  # 8-9
    elif raw_score > 0.7:  # C
        normalized_score = 7 + (raw_score - 0.7) * 10  # 7-8
    elif raw_score > 0.6:  # D
        normalized_score = 6 + (raw_score - 0.6) * 10  # 6-7
    else:  # F
        normalized_score = raw_score * 10  # 0-6
    
    # Apply quality assessment for format specifically
    if normalized_score >= 8.5:
        feedback.append("\n✨ **EXCELLENT FORMATTING**: Your resume is well-structured.")
    elif normalized_score >= 7:
        feedback.append("\n✅ **GOOD FORMATTING**: Your resume structure is decent but could be improved.")
    elif normalized_score >= 5:
        feedback.append("\n⚠️ **FORMATTING NEEDS WORK**: Your resume structure requires significant improvement.")
    else:
        feedback.append("\n❌ **POOR FORMATTING**: Consider revising your resume's entire structure and format.")
    
    return normalized_score, "\n".join(feedback)
