import json
import random
import hashlib
import logging
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from rich.logging import RichHandler
from create_intro_dataset import IntroDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

class DefinitionsDatasetGenerator:
    """Generate a fine-tuning dataset from financial definitions in text format."""
    
    def __init__(
        self,
        definitions_dir: str = "/home/zahemen/datasets/enhanced_q_and_a",
        output_file: str = "/home/zahemen/datasets/sft_datasets/financial_definitions_dataset.jsonl",
        num_samples: int = 3000  # Number of QA pairs to generate
    ):
        self.definitions_dir = Path(definitions_dir)
        self.output_file = Path(output_file)
        self.num_samples = num_samples
        self.definitions = []
        self.dataset = []
        self.intro_generator = IntroDatasetGenerator(None)  # Initialize intro generator
        self.exclude_files = {"finance_questions.txt"}
        self.max_sample_usage = 5  # Maximum times a term can be used
        self.sample_usage_counter = {}  # Track how many times each term is used
        self.min_turns = 4
        self.max_turns = 10
        
        # Question templates for definitions
        self.definition_q_templates = [
            "What is {term}?",
            "Define {term}.",
            "Can you explain what {term} means?",
            "What does {term} mean?",
            "I need to understand {term}. Can you define it?",
            "Could you provide a definition for {term}?",
            "What's the meaning of {term}?",
            "I'm unfamiliar with {term}. What is it?",
            "Please explain the term {term}.",
            "In finance, what is {term}?",
            "What does {term} refer to in finance?",
            "How would you define {term}?",
            "What's the definition of {term}?",
            "I need clarity on {term}. What does it mean?",
            "Can you tell me what {term} is?",
            "What exactly is {term}?",
            "I'm trying to understand what {term} means.",
            "Could you explain the concept of {term}?",
            "What is meant by the term {term}?",
            "For a beginner in finance, how would you explain {term}?"
        ]
        
        # Conversational question templates
        self.conversational_q_templates = [
            "I keep hearing about {term} but I don't really know what it means. Could you explain?",
            "My financial advisor mentioned {term} today and I didn't want to seem clueless. What exactly is that?",
            "I'm reading a financial article and came across '{term}' - could you break this down for me?",
            "Hey, quick question - what's {term} in simple terms?",
            "I'm confused about this whole {term} thing. Can you explain it like I'm five?",
            "So I was listening to this finance podcast and they kept talking about {term}. What is that exactly?",
            "I've been trying to improve my financial literacy and I'm stuck on understanding {term}. Help?",
            "My friend says I should know about {term} for my investments. What should I know about it?",
            "I'm new to finance and keep seeing '{term}' mentioned. What does this mean?",
            "Can you break down {term} for someone who's just starting to learn about finance?"
        ]
        
        # Follow-up questions about the same term
        self.followup_q_templates = [
            "How is {term} used in practice?",
            "Can you give me examples of {term} in real-world scenarios?",
            "Why is {term} important in finance?",
            "When would I need to be concerned with {term}?",
            "Are there different types of {term}?",
            "How does {term} affect investment decisions?",
            "What are common misconceptions about {term}?",
            "How has the concept of {term} evolved over time?",
            "What should beginners know about {term}?",
            "Who typically deals with {term} in a company?",
            "What risks are associated with {term}?",
            "How does {term} appear in financial statements?",
            "What regulations govern {term}?",
            "How do different industries approach {term}?",
            "Can {term} be optimized or improved upon?",
            "How is {term} different in international contexts?",
            "What tools or software help manage {term}?",
            "How would changes in {term} affect a company's performance?",
            "What are best practices regarding {term}?",
            "How do analysts evaluate {term}?"
        ]
        
        # Comparison templates
        self.comparison_q_templates = [
            "What's the difference between {term1} and {term2}?",
            "How do {term1} and {term2} compare?",
            "Can you explain the distinction between {term1} and {term2}?",
            "I often confuse {term1} and {term2}. How are they different?",
            "What distinguishes {term1} from {term2}?",
            "How would you compare {term1} and {term2}?",
            "I'm trying to understand the difference between {term1} and {term2}.",
            "Could you clarify how {term1} differs from {term2}?",
            "Are {term1} and {term2} related concepts? How do they differ?",
            "What are the key differences between {term1} and {term2}?",
            "How should I think about {term1} versus {term2}?",
            "In what ways are {term1} and {term2} different from each other?"
        ]
        
        # System prompts
        self.system_prompts = [
            "You are FinSight, an AI financial advisor. Provide accurate and helpful financial guidance.",
            "As FinSight, your role is to deliver expert financial insights and advice tailored to each user's needs.",
            "You are FinSight, a specialized AI designed to help users understand complex financial concepts and make informed decisions.",
            "Acting as FinSight, provide thoughtful financial guidance and explanations that help users navigate their financial questions.",
            "You are FinSight, an AI assistant specialized in financial education and advice. Provide clear and accurate information."
        ]

        # Pre-populated options for different response types
        # Example options
        self.example_options1 = ['corporate accounting', 'investment analysis', 'financial reporting', 'tax planning']
        self.example_options2 = ['a company might use', 'financial analysts often consider', 'investors typically look at']
        self.example_options3 = ['evaluating performance metrics', 'analyzing financial statements', 'making investment decisions', 'assessing risk factors']
        self.example_options4 = ['a financial manager needs to', 'an accountant is required to', 'investors want to', 'analysts are trying to']
        self.example_options5 = ['evaluate financial health', 'assess performance', 'make strategic decisions', 'report to stakeholders']
        self.example_options6 = ['When a company prepares its quarterly reports', 'During the annual auditing process', 'In merger and acquisition scenarios', 'When evaluating investment opportunities']
        self.example_options7 = ['provides insights into', 'helps quantify', 'clarifies the relationship between', 'establishes a framework for']
        self.example_options8 = ['financial performance', 'operational efficiency', 'risk exposure', 'regulatory compliance']
        
        # Importance options
        self.importance_options1 = ['provides essential information for', 'serves as a foundation for', 'enables accurate', 'plays a critical role in']
        self.importance_options2 = ['decision-making processes', 'financial analysis', 'regulatory compliance', 'risk management']
        self.importance_options3 = ['investors might make poor decisions', 'companies could face regulatory penalties', 'financial reports may be misleading', 'strategic planning would lack precision']
        self.importance_options4 = ['directly impacts', 'fundamentally affects', 'is intricately linked to', 'serves as an indicator of']
        self.importance_options5 = ['a company\'s financial health', 'investment outcomes', 'market perception', 'regulatory standing']
        self.importance_options6 = ['affects bottom-line results', 'influences strategic decisions', 'impacts compliance requirements', 'relates to fiduciary responsibilities']
        self.importance_options7 = ['success and failure', 'compliance and penalties', 'growth and stagnation', 'accurate and misleading financial reporting']
        
        # Types options
        self.types_options1 = ['standard and non-standard forms', 'direct and indirect categories', 'primary and secondary classifications', 'regulatory and operational variants']
        self.types_options2 = ['financial reporting', 'investment analysis', 'regulatory compliance', 'risk assessment']
        self.types_options3 = ['short-term and long-term', 'operating and financial', 'direct and indirect', 'primary and secondary']
        self.types_options4 = ['requires different handling', 'has different implications', 'affects financial statements differently', 'is subject to different regulations']
        self.types_options5 = ['recognized and unrecognized', 'reported and unreported', 'current and non-current', 'operating and non-operating']
        self.types_options6 = ['better financial analysis', 'more accurate reporting', 'appropriate risk assessment', 'proper regulatory compliance']
        
        # Practice options
        self.practice_options1 = ['financial statements are prepared', 'investment decisions are made', 'regulatory filings are submitted', 'risk assessments are conducted']
        self.practice_options2 = ['calculate', 'evaluate', 'report', 'analyze']
        self.practice_options3 = ['following GAAP guidelines', 'using specialized financial software', 'applying industry-standard formulas', 'consulting with subject matter experts']
        self.practice_options4 = ['systematic documentation', 'regular reporting', 'careful analysis', 'ongoing monitoring']
        self.practice_options5 = ['track', 'manage', 'report on', 'evaluate']
        self.practice_options6 = ['regulatory compliance', 'accurate financial representation', 'informed decision-making', 'proper risk management']
        self.practice_options7 = ['established procedures', 'dedicated software systems', 'specialized teams', 'regular review processes']
        self.practice_options8 = ['financial reporting is accurate', 'regulatory requirements are met', 'stakeholders receive reliable information', 'business decisions are based on sound financial principles']
        
        # Risk options
        self.risk_options1 = ['misclassification errors', 'timing discrepancies', 'valuation challenges', 'disclosure inadequacies']
        self.risk_options2 = ['financial statement misrepresentations', 'regulatory penalties', 'investor confusion', 'suboptimal business decisions']
        self.risk_options3 = ['incorrect calculation', 'inappropriate disclosure', 'inconsistent application', 'regulatory non-compliance']
        self.risk_options4 = ['rigorous internal controls', 'regular audits', 'staff training', 'clear documentation procedures']
        self.risk_options5 = ['material misstatements', 'compliance violations', 'stakeholder misunderstanding', 'operational inefficiencies']
        self.risk_options6 = ['multi-level review processes', 'specialized software solutions', 'expert consultation', 'continuous monitoring systems']
        
        # History options
        self.history_options1 = ['was much simpler', 'focused primarily on basic recording', 'lacked standardized practices', 'was not widely regulated']
        self.history_options2 = ['regulatory developments', 'accounting standard evolution', 'technological advancements', 'global financial integration']
        self.history_options3 = ['early accounting practices', 'basic business needs', 'regulatory requirements', 'financial market development']
        self.history_options4 = ['accounting standard updates', 'regulatory framework enhancements', 'industry best practice evolution', 'academic and professional research']
        self.history_options5 = ['early financial reporting needs', 'fundamental accounting principles', 'regulatory initiatives', 'market transformation events']
        self.history_options6 = ['major financial legislation', 'accounting standard revisions', 'technological innovations in reporting', 'shifts in global financial practices']
        
        # Regulation options
        self.regulation_options1 = ['GAAP principles', 'IFRS guidelines', 'SEC requirements', 'specialized industry regulations']
        self.regulation_options2 = ['properly reported', 'consistently applied', 'accurately disclosed', 'appropriately managed']
        self.regulation_options3 = ['accounting standards boards', 'securities commissions', 'industry-specific regulators', 'international oversight bodies']
        self.regulation_options4 = ['accurate disclosure', 'proper classification', 'consistent application', 'timely reporting']
        self.regulation_options5 = ['protect investors', 'ensure market transparency', 'maintain financial stability', 'promote consistency in reporting']
        self.regulation_options6 = ['disclosure requirements', 'calculation methodologies', 'documentation standards', 'reporting timelines']
        self.regulation_options7 = ['accounting standards', 'regulatory agencies', 'industry authorities', 'international conventions']
        
        # Misconception options
        self.misconception_options1 = ['is simple to calculate', 'applies uniformly across industries', 'has little impact on financial outcomes', 'can be managed without specialized knowledge']
        self.misconception_options2 = ['requires careful analysis', 'varies significantly by context', 'can substantially impact financial statements', 'demands professional expertise']
        self.misconception_options3 = ['is purely a technical accounting issue', 'doesn\'t affect business decisions', 'is identical to similar concepts', 'remains constant over time']
        self.misconception_options4 = ['reporting errors', 'compliance issues', 'ineffective financial strategies', 'missed opportunities or risks']
        self.misconception_options5 = ['related but distinct concepts', 'simpler financial measures', 'broader accounting principles', 'non-standardized practices']
        self.misconception_options6 = ['accurate financial reporting', 'proper regulatory compliance', 'effective financial analysis', 'sound business decision-making']
        
        # Industry options
        self.industry_options1 = ['banking and financial services', 'manufacturing', 'healthcare', 'technology sectors']
        self.industry_options2 = ['has specialized applications', 'faces unique regulatory requirements', 'connects to industry-specific metrics', 'presents distinct challenges']
        self.industry_options3 = ['different business models', 'specialized regulatory frameworks', 'unique operational requirements', 'varying stakeholder expectations']
        self.industry_options4 = ['financial services', 'real estate', 'energy', 'retail']
        self.industry_options5 = ['directly impacts key performance indicators', 'faces heightened regulatory scrutiny', 'requires specialized handling', 'influences strategic decisions']
        self.industry_options6 = ['healthcare', 'technology', 'manufacturing', 'telecommunications']
        self.industry_options7 = ['follow industry-specific guidelines', 'use specialized metrics', 'face unique reporting challenges', 'develop custom analytical approaches']
        
        # International options
        self.international_options1 = ['varying accounting standards', 'different regulatory frameworks', 'cultural and business practice variations', 'local financial system characteristics']
        self.international_options2 = ['IFRS guidelines', 'US GAAP standards', 'European regulatory frameworks', 'Asian market practices']
        self.international_options3 = ['different reporting requirements', 'alternative calculation methods', 'varied disclosure standards', 'distinct compliance expectations']
        self.international_options4 = ['compliance challenges for multinational organizations', 'comparison difficulties for global investors', 'complexities in cross-border transactions', 'reporting inconsistencies across markets']
        self.international_options5 = ['recognized', 'reported', 'regulated', 'interpreted']
        self.international_options6 = ['different accounting treatments', 'varying tax considerations', 'unique regulatory approaches', 'distinct reporting schedules']
        self.international_options7 = ['specialized expertise', 'localized compliance strategies', 'careful translation of financial information', 'adaptive reporting systems']

        # Conversational introductions for definitions
        self.conversational_intros = [
            "{term} is ",
            "In simple terms, {term} is ",
            "To put it simply, {term} is ",
            "Simply put, {term} is ",
            "{term}, in financial terms, is ",
            "In the financial world, {term} is ",
            "At its core, {term} is ",
            "Essentially, {term} is ",
            "{term} basically means ",
            "{term} represents ",
        ]
        
        # Transition phrases for the second term in comparisons
        self.comparison_transitions = [
            "On the other hand, {term2} is ",
            "In contrast, {term2} is ",
            "Meanwhile, {term2} is ",
            "Comparatively, {term2} is ",
            "By contrast, {term2} is ",
            "Whereas {term1} focuses on this, {term2} is ",
            "Unlike {term1}, {term2} is ",
        ]
        
        # Natural comparison conclusions
        self.comparison_conclusions = [
            "These two concepts serve different purposes in finance.",
            "Both concepts are important but apply in different financial contexts.",
            "Understanding both gives you a more complete picture of this area of finance.",
            "These concepts often work together in financial planning and analysis.",
            "Each has its own role in a comprehensive financial strategy.",
        ]
        
        # Patterns to clean up definitions
        self.cleanup_patterns = [
            # Fix redundant phrasings (case insensitive)
            (r"(?i)refers to\s+it\s+refers\s+to", "refers to"),
            (r"(?i)is\s+this\s+is", "is"),
            (r"(?i)means\s+this\s+means", "means"),
            (r"(?i)represents\s+this\s+represents", "represents"),
            (r"(?i)involves\s+this\s+involves", "involves"),
            (r"(?i)includes\s+this\s+includes", "includes"),
            (r"(?i)encompasses\s+this\s+encompasses", "encompasses"),
            # Replace formal language with conversational
            (r"(?i)is defined as", "is"),
            (r"(?i)may be defined as", "can be"),
            (r"(?i)can be defined as", "is"),
            (r"(?i)is described as", "is"),
            (r"(?i)is characterized by", "involves"),
            (r"(?i)is utilized to", "is used to"),
            (r"(?i)is employed to", "is used to"),
            (r"(?i)in nature", ""),
            # Fix awkward starts
            (r"(?i)^This refers to", "This is"),
            (r"(?i)^It refers to", "It's"),
            (r"(?i)^This is a term that", "This"),
            (r"(?i)^This is a concept that", "This"),
            (r"(?i)^This means", "This"),
            (r"(?i)^this represents", "This"),
            # Fix repeated words
            (r"(?i)\b(\w+)\s+\1\b", r"\1"),
        ]

    def extract_definitions_from_files(self) -> None:
        """Extract term-definition pairs from all text files in the directory."""
        if not self.definitions_dir.exists() or not self.definitions_dir.is_dir():
            raise FileNotFoundError(f"Definitions directory not found: {self.definitions_dir}")
        
        file_paths = [f for f in self.definitions_dir.glob("*.txt") if f.name not in self.exclude_files]
        logger.info(f"Found {len(file_paths)} definition files to process")
        
        for file_path in tqdm(file_paths, desc="Processing definition files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract term-definition pairs using a more robust pattern
                # Look for terms followed by colon and their definitions until the next term
                term_defs = self.extract_term_definitions(content)
                
                if term_defs:
                    self.definitions.extend(term_defs)
                    logger.debug(f"Extracted {len(term_defs)} definitions from {file_path.name}")
                else:
                    logger.warning(f"No definitions found in {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Extracted a total of {len(self.definitions)} definitions")
    
    def extract_term_definitions(self, content: str) -> List[Dict[str, str]]:
        """Extract term-definition pairs from text content."""
        # This is a more robust pattern to match terms and their definitions
        # It looks for lines that start with a term, followed by a colon
        # The definition continues until another line with a similar pattern is found
        
        term_defs = []
        
        # Split content by lines
        lines = content.split('\n')
        current_term = None
        current_definition = []
        
        for i, line in enumerate(lines):
            # Check if this line starts a new term definition
            term_match = re.match(r'^([^:]+):\s*(.*)', line.strip())
            
            if term_match:
                # If we already have a term being processed, save it
                if current_term is not None and current_definition:
                    # Join the definition lines and clean it up
                    definition = ' '.join([l.strip() for l in current_definition if l.strip()])
                    term_defs.append({
                        "term": current_term.strip(),
                        "definition": definition
                    })
                
                # Start a new term
                current_term = term_match.group(1)
                current_definition = [term_match.group(2)]
            elif current_term is not None:
                # Continue with current definition
                current_definition.append(line)
        
        # Don't forget the last term
        if current_term is not None and current_definition:
            definition = ' '.join([l.strip() for l in current_definition if l.strip()])
            # Clean the definition before adding
            definition = self.clean_definition_text(definition)
            term_defs.append({
                "term": current_term.strip(),
                "definition": definition
            })
        
        return term_defs

    def can_use_term(self, term: str) -> bool:
        """Check if a term can still be used based on its usage count."""
        return self.sample_usage_counter.get(term, 0) < self.max_sample_usage
    
    def mark_term_used(self, term: str) -> None:
        """Mark a term as used by incrementing its usage counter."""
        self.sample_usage_counter[term] = self.sample_usage_counter.get(term, 0) + 1
    
    def get_available_terms(self) -> List[Dict[str, str]]:
        """Get list of terms that haven't reached their usage limit."""
        return [def_dict for def_dict in self.definitions 
                if self.can_use_term(def_dict["term"])]

    def clean_definition_text(self, definition_text: str) -> str:
        """Clean definition text to make it more conversational and remove redundancies."""
        # Start with the original text
        text = definition_text.strip()
        
        # Remove common prefixes that cause redundancy when using conversational intros
        common_prefixes = [
            r"^this is\s+",
            r"^this refers to\s+",
            r"^this means\s+",
            r"^this represents\s+",
            r"^this denotes\s+",
            r"^this involves\s+",
            r"^this encompasses\s+",
            r"^this describes\s+",
            r"^it is\s+",
            r"^it refers to\s+",
            r"^it means\s+",
            r"^it represents\s+",
            r"^it denotes\s+",
            r"^it involves\s+",
            r"^it encompasses\s+",
            r"^it describes\s+",
            r"^these are\s+"
        ]
        
        # Apply prefix removal first to avoid issues with sentence start
        for prefix in common_prefixes:
            text = re.sub(prefix, "", text, flags=re.IGNORECASE)
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Fix capitalization after cleaning
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for i in range(len(sentences)):
            if sentences[i] and not sentences[i][0].isupper():
                sentences[i] = sentences[i][0].upper() + sentences[i][1:]
        
        # Rejoin sentences
        text = ' '.join(sentences)
        
        # Remove any double spaces created during cleaning
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def format_definition(self, term: str, definition_text: str) -> str:
        """Format a definition with a conversational intro."""
        # Clean the definition text first - this removes redundant prefixes
        cleaned_definition = self.clean_definition_text(definition_text)
        
        # Choose a random conversational intro
        intro_template = random.choice(self.conversational_intros)
        intro = intro_template.format(term=term)
        
        # Make sure the definition flows naturally after the intro
        # Remove redundant starts if the intro already has similar phrasing
        if intro.endswith("is "):
            cleaned_definition = re.sub(r'^is\s+', '', cleaned_definition)
            cleaned_definition = re.sub(r'^a\s+', '', cleaned_definition)
            cleaned_definition = re.sub(r'^an\s+', '', cleaned_definition)
            cleaned_definition = re.sub(r'^the\s+', '', cleaned_definition)
        elif intro.endswith("means "):
            cleaned_definition = re.sub(r'^means\s+', '', cleaned_definition)
            cleaned_definition = re.sub(r'^meaning\s+', '', cleaned_definition)
        elif intro.endswith("represents "):
            cleaned_definition = re.sub(r'^represents\s+', '', cleaned_definition)
            cleaned_definition = re.sub(r'^representing\s+', '', cleaned_definition)
        
        # Special case: check for "this is" or similar at the start after cleaning
        # This is a common issue in the source data
        cleaned_definition = re.sub(r'^this\s+is\s+', '', cleaned_definition, flags=re.IGNORECASE)
        cleaned_definition = re.sub(r'^a\s+this\s+is\s+', 'a ', cleaned_definition, flags=re.IGNORECASE)
        cleaned_definition = re.sub(r'^an\s+this\s+is\s+', 'an ', cleaned_definition, flags=re.IGNORECASE)
        cleaned_definition = re.sub(r'^the\s+this\s+is\s+', 'the ', cleaned_definition, flags=re.IGNORECASE)
        
        # Ensure first letter is lowercase if continuing a sentence
        if intro.endswith(" ") and cleaned_definition:
            cleaned_definition = cleaned_definition[0].lower() + cleaned_definition[1:]
        
        # Combine intro and definition
        formatted_definition = intro + cleaned_definition
        
        # Final check for common redundancy patterns that might have been missed
        formatted_definition = re.sub(r'is\s+this\s+is', 'is', formatted_definition, flags=re.IGNORECASE)
        formatted_definition = re.sub(r'means\s+this\s+means', 'means', formatted_definition, flags=re.IGNORECASE)
        formatted_definition = re.sub(r'represents\s+this\s+represents', 'represents', formatted_definition, flags=re.IGNORECASE)
        
        return formatted_definition

    def create_qa_pair(self, definition: Dict[str, str], with_conversation: bool = False) -> Dict[str, Any]:
        """Create a QA pair from a definition."""
        term = definition["term"]
        definition_text = definition["definition"]
        
        # Generate a question from the template
        if random.random() < 0.5:
            question_templates = self.conversational_q_templates
        else:
            question_templates = self.definition_q_templates
            
        question = random.choice(question_templates).format(term=term)
        
        # Create unique conversation ID
        conv_id = hashlib.sha256(f"{term}_{question}".encode()).hexdigest()
        
        # Format the definition with a conversational intro
        response = self.format_definition(term, definition_text)
        
        # Basic Q&A format (directly to the question)
        if not with_conversation:
            messages = [
                {
                    "role": "system",
                    "content": random.choice(self.system_prompts)
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
            
            # Mark this term as used
            self.mark_term_used(term)
            
            return {
                "messages": messages,
                "metadata": {
                    "source": "financial_definitions",
                    "term": term,
                    "conversation_id": conv_id,
                    "type": "definition"
                }
            }
        
        # Extended multi-turn conversation format
        else:
            # Start with system and main question
            messages = [
                {
                    "role": "system", 
                    "content": random.choice(self.system_prompts)
                },
                {
                    "role": "user", 
                    "content": question
                },
                {
                    "role": "assistant", 
                    "content": response
                }
            ]
            
            # Add follow-up turns
            num_turns = random.randint(self.min_turns - 1, self.max_turns - 1)  # -1 because we already have one turn
            
            # Get a list of potential follow-up questions to avoid repetition
            followup_questions = random.sample(
                self.followup_q_templates, 
                min(num_turns, len(self.followup_q_templates))
            )
            
            for i in range(num_turns):
                # Generate a follow-up question about the same term
                if i < len(followup_questions):
                    followup_q = followup_questions[i].format(term=term)
                else:
                    # If we've used all pre-defined questions, create a variation
                    base_q = random.choice(self.followup_q_templates)
                    followup_q = f"{base_q.format(term=term)} In particular, how does this relate to financial reporting?"
                
                # Generate a contextual follow-up response with proper formatting
                followup_response = self.generate_contextual_response(term, followup_q, definition_text)
                
                messages.extend([
                    {"role": "user", "content": followup_q},
                    {"role": "assistant", "content": followup_response}
                ])
            
            # Mark this term as used
            self.mark_term_used(term)
            
            return {
                "messages": messages,
                "metadata": {
                    "source": "financial_definitions",
                    "term": term,
                    "conversation_id": conv_id,
                    "type": "multi_turn",
                    "turns": len(messages) // 2  # Count actual turns
                }
            }

    def generate_contextual_response(self, term: str, question: str, original_definition: str) -> str:
        """Generate a contextual response to a follow-up question based on the original definition."""
        # Extract key phrases from the original definition
        original_sentences = re.split(r'[.!?]', original_definition)
        original_sentences = [s.strip() for s in original_sentences if s.strip()]
        
        # Keyword patterns to look for in the question
        question_patterns = {
            'example': ['example', 'instance', 'case', 'illustration', 'scenario'],
            'importance': ['important', 'significance', 'relevance', 'why', 'matter'],
            'types': ['types', 'kinds', 'varieties', 'different', 'categories', 'classifications'],
            'practice': ['practice', 'real-world', 'application', 'applied', 'use', 'implement'],
            'risk': ['risk', 'danger', 'concern', 'problem', 'issue', 'challenge'],
            'history': ['history', 'evolution', 'developed', 'origin', 'background'],
            'regulation': ['regulation', 'rule', 'law', 'compliance', 'govern', 'legal'],
            'misconception': ['misconception', 'mistake', 'wrongly', 'incorrect', 'misunderstood'],
            'industry': ['industry', 'sector', 'field', 'business', 'companies'],
            'international': ['international', 'global', 'country', 'worldwide', 'foreign'],
        }
        
        # Check which category the question falls into
        question_lower = question.lower()
        detected_categories = []
        
        for category, keywords in question_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_categories.append(category)
        
        # If no specific category detected, choose a random one
        if not detected_categories:
            detected_categories = [random.choice(list(question_patterns.keys()))]
        
        # Select a template based on the detected category
        primary_category = detected_categories[0]
        
        # Use the template without lambda functions to avoid JSON serialization issues
        templates = {
            'example': [
                f"In practice, {term} is commonly used in financial contexts such as {random.choice(self.example_options1)}. {original_sentences[0] if original_sentences else ''} For example, {random.choice(self.example_options2)} {term} when {random.choice(self.example_options3)}.",
                
                f"{term} is frequently applied in real-world scenarios. {original_sentences[0] if original_sentences else ''} A typical example would be when {random.choice(self.example_options4)} {random.choice(self.example_options5)}.",
                
                f"Let me illustrate {term} with a practical example: {random.choice(self.example_options6)}, {term} becomes crucial because it {random.choice(self.example_options7)} {random.choice(self.example_options8)}."
            ],
            'importance': [
                f"{term} is important in finance because it {random.choice(self.importance_options1)} {random.choice(self.importance_options2)}. {original_sentences[0] if original_sentences else ''} Without proper understanding of {term}, {random.choice(self.importance_options3)}.",
                
                f"The significance of {term} cannot be overstated in the financial world. It {random.choice(self.importance_options4)} {random.choice(self.importance_options5)}. {original_sentences[0] if original_sentences else ''} This is why financial professionals place such emphasis on understanding and correctly applying this concept.",
                
                f"Understanding {term} matters because it {random.choice(self.importance_options6)}. {original_sentences[0] if original_sentences else ''} In today's complex financial environment, proper handling of {term} can be the difference between {random.choice(self.importance_options7)}."
            ],
            'types': [
                f"There are several types of {term} that financial professionals should be familiar with. These include {random.choice(self.types_options1)}. {original_sentences[0] if original_sentences else ''} Each type serves a specific purpose in {random.choice(self.types_options2)}.",
                
                f"{term} can be categorized into different types such as {random.choice(self.types_options3)}. {original_sentences[0] if original_sentences else ''} The distinction is important because each category {random.choice(self.types_options4)}.",
                
                f"Financial experts typically classify {term} into several categories including {random.choice(self.types_options5)}. {original_sentences[0] if original_sentences else ''} This classification helps in {random.choice(self.types_options6)}."
            ],
            'practice': [
                f"In practice, {term} is applied when {random.choice(self.practice_options1)}. {original_sentences[0] if original_sentences else ''} Financial professionals typically {random.choice(self.practice_options2)} {term} by {random.choice(self.practice_options3)}.",
                
                f"The practical application of {term} involves {random.choice(self.practice_options4)}. {original_sentences[0] if original_sentences else ''} In day-to-day operations, financial teams {random.choice(self.practice_options5)} {term} to ensure {random.choice(self.practice_options6)}.",
                
                f"Organizations implement {term} through {random.choice(self.practice_options7)}. {original_sentences[0] if original_sentences else ''} This practical implementation ensures that {random.choice(self.practice_options8)}."
            ],
            'risk': [
                f"Several risks are associated with {term}, including {random.choice(self.risk_options1)}. {original_sentences[0] if original_sentences else ''} These risks can lead to {random.choice(self.risk_options2)} if not properly managed.",
                
                f"The main risks related to {term} involve {random.choice(self.risk_options3)}. {original_sentences[0] if original_sentences else ''} Financial professionals mitigate these risks through {random.choice(self.risk_options4)}.",
                
                f"When dealing with {term}, organizations face risks such as {random.choice(self.risk_options5)}. {original_sentences[0] if original_sentences else ''} Effective risk management for {term} typically includes {random.choice(self.risk_options6)}."
            ],
            'history': [
                f"The concept of {term} has evolved significantly over time. Originally, it {random.choice(self.history_options1)}. {original_sentences[0] if original_sentences else ''} Modern understanding of {term} has been shaped by {random.choice(self.history_options2)}.",
                
                f"Historically, {term} emerged from {random.choice(self.history_options3)}. {original_sentences[0] if original_sentences else ''} Over decades, the concept has been refined through {random.choice(self.history_options4)}.",
                
                f"The history of {term} can be traced back to {random.choice(self.history_options5)}. {original_sentences[0] if original_sentences else ''} Key developments in its evolution include {random.choice(self.history_options6)}."
            ],
            'regulation': [
                f"{term} is governed by several regulations and standards, including {random.choice(self.regulation_options1)}. {original_sentences[0] if original_sentences else ''} These regulatory frameworks ensure that {term} is {random.choice(self.regulation_options2)} across organizations.",
                
                f"The regulatory landscape for {term} includes {random.choice(self.regulation_options3)}. {original_sentences[0] if original_sentences else ''} Compliance requirements typically focus on {random.choice(self.regulation_options4)} of {term} in financial statements.",
                
                f"Regulations concerning {term} are designed to {random.choice(self.regulation_options5)}. {original_sentences[0] if original_sentences else ''} Organizations must comply with {random.choice(self.regulation_options6)} established by {random.choice(self.regulation_options7)}."
            ],
            'misconception': [
                f"A common misconception about {term} is that it {random.choice(self.misconception_options1)}. {original_sentences[0] if original_sentences else ''} In reality, {term} {random.choice(self.misconception_options2)}.",
                
                f"Many people mistakenly believe that {term} {random.choice(self.misconception_options3)}. {original_sentences[0] if original_sentences else ''} This misunderstanding can lead to {random.choice(self.misconception_options4)}.",
                
                f"One widespread misconception is confusing {term} with {random.choice(self.misconception_options5)}. {original_sentences[0] if original_sentences else ''} Understanding the precise definition and application of {term} is crucial for {random.choice(self.misconception_options6)}."
            ],
            'industry': [
                f"Different industries approach {term} in varying ways. For example, in {random.choice(self.industry_options1)}, {term} typically {random.choice(self.industry_options2)}. {original_sentences[0] if original_sentences else ''} These industry variations reflect {random.choice(self.industry_options3)}.",
                
                f"{term} has particular significance in industries such as {random.choice(self.industry_options4)} where it {random.choice(self.industry_options5)}. {original_sentences[0] if original_sentences else ''} Industry professionals develop specialized expertise to manage {term} in their specific context.",
                
                f"The application of {term} varies significantly across different sectors. In {random.choice(self.industry_options6)}, organizations typically {random.choice(self.industry_options7)}. {original_sentences[0] if original_sentences else ''} These sectoral differences reflect the diverse ways {term} manifests in different business environments."
            ],
            'international': [
                f"{term} is treated differently across international boundaries due to {random.choice(self.international_options1)}. {original_sentences[0] if original_sentences else ''} For example, while {random.choice(self.international_options2)} approach {term} one way, other regions may have {random.choice(self.international_options3)}.",
                
                f"International variations in handling {term} create {random.choice(self.international_options4)}. {original_sentences[0] if original_sentences else ''} Financial professionals working globally must understand how {term} is {random.choice(self.international_options5)} in different jurisdictions.",
                
                f"Across different countries, {term} may be subject to {random.choice(self.international_options6)}. {original_sentences[0] if original_sentences else ''} These international differences require {random.choice(self.international_options7)} for organizations operating in multiple jurisdictions."
            ],
        }
        
        # If we don't have a template for the detected category, use a general one
        if primary_category not in templates:
            response = f"Regarding {term}, {original_sentences[0] if original_sentences else ''} This concept is essential in financial contexts and helps professionals better understand and manage financial information. It provides structure and clarity to what might otherwise be complex financial scenarios."
        else:
            # Select a template from the appropriate category
            response = random.choice(templates[primary_category])
        
        # Clean the response to make it more conversational
        response = self.clean_definition_text(response)
        
        return response

    def create_comparison_qa_pair(self, term1_def: Dict[str, str], term2_def: Dict[str, str]) -> Dict[str, Any]:
        """Create a QA pair comparing two financial terms."""
        term1 = term1_def["term"]
        term2 = term2_def["term"]
        definition1 = term1_def["definition"]
        definition2 = term2_def["definition"]
        
        # Generate a comparison question
        question = random.choice(self.comparison_q_templates).format(term1=term1, term2=term2)
        
        # Create unique conversation ID
        conv_id = hashlib.sha256(f"{term1}_{term2}_{question}".encode()).hexdigest()
        
        # Create a comparison response
        comparison_response = self.create_comparison_response(term1, term2, definition1, definition2)
        
        if random.random() < 0.5:  # 50% chance for a multi-turn conversation
            # Create a basic conversation with follow-up
            num_turns = random.randint(1, 3)  # 1-3 additional turns after the initial response
            
            # Start with the main question and answer
            messages = [
                {
                    "role": "system",
                    "content": random.choice(self.system_prompts)
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": comparison_response
                }
            ]
            
            # Follow-up questions for comparisons
            followup_templates = [
                "When would I use {term1} instead of {term2}?",
                "Can you give me an example where both {term1} and {term2} would be used together?",
                "Which is more important to understand, {term1} or {term2}?",
                "How do professionals use {term1} and {term2} in their analysis?",
                "Are there any situations where {term1} and {term2} overlap?",
                "What are the practical implications of choosing {term1} over {term2}?"
            ]
            
            for _ in range(num_turns):
                followup_q = random.choice(followup_templates).format(term1=term1, term2=term2)
                followup_response = self.generate_contextual_response(term1, followup_q, definition1)
                
                messages.extend([
                    {"role": "user", "content": followup_q},
                    {"role": "assistant", "content": followup_response}
                ])
            
            # Mark these terms as used
            self.mark_term_used(term1)
            self.mark_term_used(term2)
            
            return {
                "messages": messages,
                "metadata": {
                    "source": "financial_definitions",
                    "terms": f"{term1}, {term2}",
                    "conversation_id": conv_id,
                    "type": "comparison",
                    "turns": len(messages) // 2  # Count actual turns
                }
            }
        
        # Single-turn comparison format
        else:
            messages = [
                {
                    "role": "system",
                    "content": random.choice(self.system_prompts)
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": comparison_response
                }
            ]
            
            # Mark these terms as used
            self.mark_term_used(term1)
            self.mark_term_used(term2)
            
            return {
                "messages": messages,
                "metadata": {
                    "source": "financial_definitions",
                    "terms": f"{term1}, {term2}",
                    "conversation_id": conv_id,
                    "type": "comparison"
                }
            }

    def create_comparison_response(self, term1: str, term2: str, definition1: str, definition2: str) -> str:
        """Create a response comparing two financial terms."""
        # First completely clean the definitions to remove redundant phrasing
        clean_def1 = self.clean_definition_text(definition1)
        clean_def2 = self.clean_definition_text(definition2)
        
        # Format first term
        term1_intro = random.choice(self.conversational_intros).format(term=term1)
        
        # Extra checks for "this is" pattern that might appear in different forms
        clean_def1 = re.sub(r'^this\s+is\s+', '', clean_def1, flags=re.IGNORECASE)
        clean_def1 = re.sub(r'^a\s+this\s+is\s+', 'a ', clean_def1, flags=re.IGNORECASE)
        
        if term1_intro.endswith("is "):
            clean_def1 = re.sub(r'^is\s+', '', clean_def1)
            clean_def1 = re.sub(r'^a\s+', '', clean_def1)
            clean_def1 = re.sub(r'^an\s+', '', clean_def1)
            clean_def1 = re.sub(r'^the\s+', '', clean_def1)
        
        if term1_intro.endswith(" ") and clean_def1:
            clean_def1 = clean_def1[0].lower() + clean_def1[1:]
        
        term1_part = term1_intro + clean_def1
        
        # Format second term with transition
        transition = random.choice(self.comparison_transitions).format(term1=term1, term2=term2)
        
        # Extra checks for "this is" pattern
        clean_def2 = re.sub(r'^this\s+is\s+', '', clean_def2, flags=re.IGNORECASE)
        clean_def2 = re.sub(r'^a\s+this\s+is\s+', 'a ', clean_def2, flags=re.IGNORECASE)
        
        if transition.endswith("is "):
            clean_def2 = re.sub(r'^is\s+', '', clean_def2)
            clean_def2 = re.sub(r'^a\s+', '', clean_def2)
            clean_def2 = re.sub(r'^an\s+', '', clean_def2)
            clean_def2 = re.sub(r'^the\s+', '', clean_def2)
        
        if transition.endswith(" ") and clean_def2:
            clean_def2 = clean_def2[0].lower() + clean_def2[1:]
        
        term2_part = transition + clean_def2
        
        # Final check for common redundancy patterns
        term1_part = re.sub(r'is\s+this\s+is', 'is', term1_part, flags=re.IGNORECASE)
        term2_part = re.sub(r'is\s+this\s+is', 'is', term2_part, flags=re.IGNORECASE)
        
        # Choose a format based on the length of definitions
        if len(clean_def1) + len(clean_def2) < 300:
            # Shorter combined format
            response = (
                f"Let me explain the difference between {term1} and {term2}.\n\n"
                f"{term1_part}\n\n"
                f"{term2_part}"
            )
        else:
            # Bulleted format for longer definitions
            response = (
                f"Here's how {term1} and {term2} compare:\n\n"
                f"• {term1}: {clean_def1}\n\n"
                f"• {term2}: {clean_def2}"
            )
        
        # Add a conclusion if it makes sense (30% chance to avoid being too wordy)
        if random.random() < 0.3:
            response += f"\n\n{random.choice(self.comparison_conclusions)}"
            
        return response

    def generate_dataset(self) -> None:
        """Generate the dataset with definition QA pairs."""
        if not self.definitions:
            self.extract_definitions_from_files()
            
        # Calculate number of samples for each type
        num_comparisons = int(self.num_samples * 0.25)
        num_remaining = self.num_samples - num_comparisons
        num_with_conversation = int(num_remaining * 0.6)  # 60% with extended conversation
        num_simple = num_remaining - num_with_conversation  # 40% with simple Q&A
        
        logger.info(f"Generating {num_simple} simple Q&A pairs")
        
        # Generate samples with replacement if we need more samples than definitions
        for _ in tqdm(range(num_simple), desc="Generating simple Q&A pairs"):
            available_terms = self.get_available_terms()
            if not available_terms:
                logger.warning("No more available terms to use for simple Q&A pairs")
                break
            definition = random.choice(available_terms)
            qa_pair = self.create_qa_pair(definition, with_conversation=False)
            self.dataset.append(qa_pair)
            
        logger.info(f"Generating {num_with_conversation} conversation-style Q&A pairs")
        
        for _ in tqdm(range(num_with_conversation), desc="Generating conversation Q&A pairs"):
            available_terms = self.get_available_terms()
            if not available_terms:
                logger.warning("No more available terms to use for conversation Q&A pairs")
                break
            definition = random.choice(available_terms)
            qa_pair = self.create_qa_pair(definition, with_conversation=True)
            self.dataset.append(qa_pair)
            
        logger.info(f"Generating {num_comparisons} comparison Q&A pairs")
        
        for _ in tqdm(range(num_comparisons), desc="Generating comparison Q&A pairs"):
            available_terms = self.get_available_terms()
            if len(available_terms) < 2:
                logger.warning("Not enough available terms to create comparison Q&A pairs")
                break
            
            # Pick two different random terms
            term1_def = random.choice(available_terms)
            term2_def = random.choice(available_terms)
            
            # Ensure we have two different terms
            attempts = 0
            while term1_def["term"] == term2_def["term"] and attempts < 10:
                term2_def = random.choice(available_terms)
                attempts += 1
            
            # Skip if we couldn't find two different terms
            if term1_def["term"] == term2_def["term"]:
                continue
                
            qa_pair = self.create_comparison_qa_pair(term1_def, term2_def)
            self.dataset.append(qa_pair)
            
        logger.info(f"Generated {len(self.dataset)} total samples")

    def save_dataset(self) -> None:
        """Save the generated dataset to a JSONL file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for sample in self.dataset:
                f.write(json.dumps(sample) + '\n')
                
        logger.info(f"Saved dataset to {self.output_file}")

    def run(self) -> None:
        """Execute the full dataset generation process."""
        logger.info("Starting financial definitions dataset generation")
        self.extract_definitions_from_files()
        self.generate_dataset()
        self.save_dataset()
        logger.info("Dataset generation complete!")
        
        # Print a sample from the dataset
        if self.dataset:
            sample = random.choice(self.dataset)
            logger.info("\nSample from generated dataset:")
            for msg in sample["messages"]:
                if msg["role"] == "system":
                    logger.info(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    logger.info(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    logger.info(f"Assistant: {msg['content']}")
            logger.info(f"\nMetadata: {sample['metadata']}")

if __name__ == "__main__":
    # First, extract definitions if needed
    from extract_financial_definitions import FinancialDefinitionsExtractor
    
    # Check if the definitions file exists
    if not Path("/home/zahemen/datasets/financial_definitions.jsonl").exists():
        logger.info("Definitions file not found. Extracting definitions first...")
        extractor = FinancialDefinitionsExtractor()
        extractor.run()
    
    # Generate the dataset
    generator = DefinitionsDatasetGenerator()
    generator.run()
