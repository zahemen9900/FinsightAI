import json
import random
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any
import re
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
    """Generate a fine-tuning dataset from financial definitions."""
    
    def __init__(
        self,
        definitions_file: str = "/home/zahemen/datasets/financial_definitions.jsonl",
        output_file: str = "/home/zahemen/datasets/financial_definitions_dataset.jsonl",
        num_samples: int = 3000  # Number of QA pairs to generate
    ):
        self.definitions_file = Path(definitions_file)
        self.output_file = Path(output_file)
        self.num_samples = num_samples
        self.definitions = []
        self.dataset = []
        self.intro_generator = IntroDatasetGenerator(None)  # Initialize intro generator
        
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
        
        # Add more natural, conversational questions
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
            "Can you break down {term} for someone who's just starting to learn about finance?",
            "I've heard '{term}' thrown around at work but I'm too embarrassed to ask what it means. Can you explain?",
            "So... what exactly is {term}? I never really understood it properly.",
            "I'm trying to explain {term} to my partner but I'm not sure I understand it myself. Can you help?",
            "This might be a basic question, but what is {term} and why is it important?",
            "My financial planner keeps mentioning {term} and I just nod along. What does it actually mean?",
            "I saw {term} in a financial newsletter and realized I have no idea what it means. Can you explain?",
            "I've been pretending to know what {term} is for too long. Please explain it to me!",
            "Everyone in my investment group talks about {term} and I'm too afraid to ask what it means now. Help?",
            "I'm supposed to know what {term} is at this point in my career, but I don't. Can you explain without judging?",
            "Can you tell me about {term} in everyday language, not financial jargon?"
        ]
        
        # System prompts
        self.system_prompts = [
            "You are FinSight, an AI financial advisor. Provide accurate and helpful financial guidance.",
            "As FinSight, your role is to deliver expert financial insights and advice tailored to each user's needs.",
            "You are FinSight, a specialized AI designed to help users understand complex financial concepts and make informed decisions.",
            "Acting as FinSight, provide thoughtful financial guidance and explanations that help users navigate their financial questions.",
            "You are FinSight, an AI assistant specialized in financial education and advice. Provide clear and accurate information."
        ]
        
        # Follow-up questions to add at the end of responses
        self.followup_questions = [
            "Is there a specific aspect of {term} you'd like to understand better?",
            "Would you like me to explain how {term} relates to other financial concepts?",
            "Do you have any specific questions about how {term} applies to your situation?",
            "Is there anything else you'd like to know about {term} or related concepts?",
            "How familiar are you with {term} in practice?",
            "Would you like some examples of how {term} works in real-world scenarios?",
            "Are you asking about {term} for personal finance or business purposes?",
            "Is there a specific financial goal you're trying to achieve related to {term}?",
            "Would it help if I explained some related terms to {term}?",
            "Do you need any clarification on the definition of {term}?"
        ]
        
        # More conversational follow-up questions
        self.conversational_followup_questions = [
            "Does that make sense, or should I try explaining {term} differently?",
            "Have you encountered {term} in your own financial situation before?",
            "What made you curious about {term} specifically?",
            "Do you think {term} might be relevant to your current financial goals?",
            "Was there a specific reason you wanted to understand {term}?",
            "Does knowing about {term} help with what you were trying to figure out?",
            "Was that explanation of {term} clear enough, or would you like me to go deeper?",
            "How do you think understanding {term} might help with your financial decisions?",
            "Does {term} relate to something specific you're working on?",
            "Was there something specific about {term} that confused you?",
            "Now that you understand {term}, are there other related concepts you're curious about?",
            "Do you have any practical situations where you might use this understanding of {term}?",
            "What other financial terms have you been wondering about besides {term}?",
            "Would you like to know how {term} might affect your personal finances?",
            "Is {term} something you're dealing with right now in your financial life?"
        ]
        
        # Conversation starters
        self.conversation_starters = [
            {
                "user": "Hi there!",
                "assistant": "Hello! I'm FinSight, your AI financial advisor. I can help with financial definitions, investment strategies, retirement planning, and more. How can I assist you today?"
            },
            {
                "user": "Hello, I need some financial advice.",
                "assistant": "Hi there! I'm FinSight, here to help with your financial questions. Whether you need explanations of financial terms, investment guidance, or planning strategies, I'm ready to assist. What would you like to know?"
            },
            {
                "user": "Can you help me understand financial terms?",
                "assistant": "Absolutely! As FinSight, I specialize in explaining financial concepts clearly. Financial terminology can be complex, but I'm here to break it down for you. What specific term or concept would you like me to explain?"
            },
            {
                "user": "I'm confused about some financial concepts.",
                "assistant": "I understand financial terminology can be overwhelming! I'm FinSight, and I'm here to help demystify financial concepts for you. I can provide clear explanations of terms, principles, or strategies. What specific concept is giving you trouble?"
            },
            {
                "user": "I need to learn more about financial terminology.",
                "assistant": "You've come to the right place! I'm FinSight, and I'm designed to help people understand financial concepts. Financial literacy is important, and I'm happy to explain any terms or concepts you're curious about. What would you like to learn first?"
            }
        ]
        
        # Add comparison question templates
        self.comparison_q_templates = [
            "What's the difference between {term1} and {term2}?",
            "Can you explain how {term1} differs from {term2}?",
            "How do {term1} and {term2} compare to each other?",
            "What distinguishes {term1} from {term2}?",
            "Could you explain the difference between {term1} and {term2}?",
            "I'm confused about {term1} and {term2}. What's the difference?",
            "How would you differentiate between {term1} and {term2}?",
            "Are {term1} and {term2} the same thing or different concepts?",
            "I often mix up {term1} and {term2}. Can you clarify the distinction?",
            "What are the key differences between {term1} and {term2}?",
            "In finance, how does {term1} differ from {term2}?",
            "I need to understand the contrast between {term1} and {term2}."
        ]
        
        # Add conversational comparison question templates
        self.conversational_comparison_templates = [
            "I've always been confused about {term1} vs {term2} - can you clarify?",
            "My colleague mentioned both {term1} and {term2} and I'm not sure how they differ. Can you explain?",
            "I'm reading about both {term1} and {term2} in this article but can't figure out the difference.",
            "Could you break down the difference between {term1} and {term2} in simple terms?",
            "I always get {term1} and {term2} mixed up. How do they differ?",
            "My financial advisor mentioned both {term1} and {term2}, but I'm too embarrassed to ask the difference.",
            "When should I be concerned with {term1} vs {term2}? What's the distinction?",
            "In my finance class, we discussed both {term1} and {term2}. Could you explain the difference?",
            "Are {term1} and {term2} related concepts? How do they differ exactly?",
            "What's the relationship between {term1} and {term2}? Are they opposites or complementary?"
        ]
        
        # Percentage of comparison questions to generate (15% of the total)
        self.comparison_percentage = 0.25

    def load_definitions(self) -> None:
        """Load financial definitions from JSONL file."""
        if not self.definitions_file.exists():
            raise FileNotFoundError(f"Definitions file not found: {self.definitions_file}")
            
        try:
            self.definitions = []
            with open(self.definitions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    definition = json.loads(line)
                    # Only include if both term and definition are non-empty
                    if definition.get('term') and definition.get('definition'):
                        self.definitions.append(definition)
                        
            logger.info(f"Loaded {len(self.definitions)} definitions")
            
            if len(self.definitions) == 0:
                raise ValueError("No valid definitions found in the input file")
                
        except Exception as e:
            logger.error(f"Error loading definitions: {e}")
            raise

    def enhance_definition(self, term: str, definition: str) -> str:
        """Enhance a definition to make it more conversational and informative."""
        # Add a random starter phrase occasionally
        starter_phrases = [
            f"In finance, ",
            f"Simply put, ",
            f"Essentially, ",
            f"In financial contexts, "
        ]
        
        # More conversational starter phrases
        conversational_starters = [
            f"Let me explain what {term} means. ",
            f"I'd be happy to explain {term}. ",
            f"Great question about {term}. ",
            f"I can definitely help you understand {term}. ",
            f"Thanks for asking about {term}. ",
            f"That's a good question. {term} is ",
            f"I'm glad you asked about {term}. ",
            f"{term} is an important concept in finance. Essentially, it's ",
            f"Many people ask about {term}. In simple terms, it's ",
            f"Don't worry, many people aren't familiar with {term}. It's "
        ]
        
        # Add examples or additional context occasionally
        example_phrases = [
            f" For example, when analyzing a company's financial statements, {term.lower()} is an important factor to consider.",
            f" This concept is particularly important in investment analysis and financial planning.",
            f" Understanding {term.lower()} is crucial for making informed financial decisions.",
            f" This is a fundamental concept in accounting and financial reporting.",
            f" This term is commonly used in discussions about corporate finance and investment strategy."
        ]
        
        # More conversational example phrases
        conversational_examples = [
            f" To give you a real-world example, think about when you're {random.choice(['reviewing your investments', 'planning for retirement', 'looking at a company report', 'talking to a financial advisor'])} - that's where {term.lower()} comes into play.",
            f" If you've ever {random.choice(['looked at investment options', 'checked your retirement account', 'reviewed a financial statement', 'wondered about market trends'])}, you've probably encountered {term.lower()} without realizing it.",
            f" In everyday terms, it's like {random.choice(['saving for a rainy day', 'comparing prices before buying', 'understanding the value of what you own', 'planning ahead for expenses'])} but in the financial world.",
            f" Many people use {term.lower()} when they're trying to {random.choice(['make smarter investment choices', 'understand market movements', 'plan their financial future', 'evaluate a company health'])}.",
            f" You might encounter {term.lower()} when you're {random.choice(['reading financial news', 'checking your investment portfolio', 'talking with a financial planner', 'researching companies to invest in'])}."
        ]
        
        enhanced = definition
        
        # 40% chance to add a starter phrase if definition doesn't already start with the term
        if random.random() < 0.4 and not definition.lower().startswith(term.lower()):
            # 50% chance to use conversational starter instead of traditional one
            if random.random() < 0.5:
                enhanced = random.choice(conversational_starters) + enhanced
            else:
                enhanced = random.choice(starter_phrases) + enhanced
        
        # 40% chance to add an example or additional context
        if random.random() < 0.4:
            # 50% chance to use conversational example
            if random.random() < 0.5:
                enhanced += random.choice(conversational_examples)
            else:
                enhanced += random.choice(example_phrases)
        
        # 40% chance to add a follow-up question
        if random.random() < 0.4:
            # 50% chance to use conversational follow-up
            if random.random() < 0.5:
                followup = random.choice(self.conversational_followup_questions).format(term=term)
            else:
                followup = random.choice(self.followup_questions).format(term=term)
            enhanced += f"\n\n{followup}"
        
        return enhanced

    def create_comparison_qa_pair(self, term1_def: Dict[str, str], term2_def: Dict[str, str]) -> Dict[str, Any]:
        """Create a QA pair for comparing two financial terms."""
        term1 = term1_def["term"]
        term2 = term2_def["term"]
        definition1 = term1_def["definition"]
        definition2 = term2_def["definition"]
        
        # Generate a comparison question - 50% chance to use conversational template
        if random.random() < 0.5:
            question = random.choice(self.conversational_comparison_templates).format(term1=term1, term2=term2)
        else:
            question = random.choice(self.comparison_q_templates).format(term1=term1, term2=term2)
        
        # Create a comparison answer
        answer = self.create_comparison_response(term1, term2, definition1, definition2)
        
        # Create unique conversation ID
        conv_id = hashlib.sha256(f"{term1}_{term2}_{question}".encode()).hexdigest()
        
        # 70% chance to create a direct Q&A pair, 30% chance to create a conversation
        if random.random() < 0.7:
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
                    "content": answer
                }
            ]
        else:
            # Use conversation starter
            starter = random.choice(self.conversation_starters)
            
            # 50% chance to use intro_generator 
            if random.random() < 0.5:
                messages = self.intro_generator.generate_conversation()
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
            else:
                messages = [
                    {
                        "role": "system",
                        "content": random.choice(self.system_prompts)
                    },
                    {
                        "role": "user", 
                        "content": starter["user"]
                    },
                    {
                        "role": "assistant",
                        "content": starter["assistant"]
                    },
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            
            # 30% chance to add a follow-up question
            if random.random() < 0.3:
                followup_questions = [
                    f"Which is more important to understand, {term1} or {term2}?",
                    f"In which situations would {term1} be more relevant than {term2}?",
                    f"Can you give me an example where both {term1} and {term2} would be used together?",
                    f"Should I focus more on {term1} or {term2} for my financial education?",
                    f"Are there any other concepts related to both {term1} and {term2} that I should know about?",
                    f"How do professional investors use the concepts of {term1} and {term2}?"
                ]
                
                user_followup = random.choice(followup_questions)
                
                # Generate follow-up response based on the terms
                assistant_followup = self.create_followup_comparison_response(term1, term2)
                
                messages.append({"role": "user", "content": user_followup})
                messages.append({"role": "assistant", "content": assistant_followup})
        
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
        # Strip any existing term references from the definition
        def clean_term_from_def(term, definition):
            # Remove phrases like "Term is defined as" or "Term refers to"
            patterns = [
                f"^{re.escape(term)}\\s+is\\s+",
                f"^{re.escape(term)}\\s+refers\\s+to\\s+",
                f"^{re.escape(term)}\\s+means\\s+",
                f"^In\\s+finance,\\s+{re.escape(term)}\\s+is\\s+"
            ]
            result = definition
            for pattern in patterns:
                result = re.sub(pattern, "", result, flags=re.IGNORECASE)
            return result
        
        cleaned_def1 = clean_term_from_def(term1, definition1)
        cleaned_def2 = clean_term_from_def(term2, definition2)
        
        # Formats for comparison responses
        formats = [
            # Standard comparison format
            f"Let me explain the difference between {term1} and {term2}.\n\n"
            f"{cleaned_def1}\n\n"
            f"On the other hand, {term2} refers to {cleaned_def2}\n\n"
            f"The key distinction is that {term1} focuses on {random.choice(['specific aspects', 'certain elements', 'particular characteristics'])} "
            f"of financial {random.choice(['operations', 'concepts', 'principles'])}, while {term2} is more concerned with "
            f"{random.choice(['different components', 'other factors', 'alternative considerations'])}.",
            
            # Table-like format
            f"The difference between {term1} and {term2} can be understood as follows:\n\n"
            f"{term1}:\n- {cleaned_def1}\n\n"
            f"{term2}:\n- {cleaned_def2}\n\n"
            f"While both are important financial concepts, they serve different purposes in the financial ecosystem.",
            
            # Compare and contrast format
            f"{term1} and {term2} are distinct financial concepts:\n\n"
            f"{term1}: {cleaned_def1}\n\n"
            f"{term2}: {cleaned_def2}\n\n"
            f"Understanding both concepts is important for comprehensive financial knowledge, as they represent different aspects of "
            f"{random.choice(['financial markets', 'economic principles', 'investment strategies', 'financial analysis'])}.",
            
            # Explanation with practical note
            f"These two terms represent different financial concepts:\n\n"
            f"{cleaned_def1}\n\n"
            f"Meanwhile, {term2} is {cleaned_def2}\n\n"
            f"In practical terms, you might encounter {term1} when dealing with {random.choice(['investment decisions', 'financial planning', 'market analysis', 'economic forecasting'])}, "
            f"whereas {term2} is more relevant in contexts involving {random.choice(['risk assessment', 'portfolio management', 'financial reporting', 'strategic planning'])}."
        ]
        
        # Select a format and add a follow-up question sometimes
        response = random.choice(formats)
        if random.random() < 0.3:
            followup = random.choice([
                f"Would you like me to elaborate on either {term1} or {term2} in more detail?",
                f"Is there a specific aspect of either {term1} or {term2} you'd like me to explain further?",
                f"Do you have any specific questions about how {term1} or {term2} might apply to your financial situation?",
                f"Is there another financial concept you'd like me to compare with either {term1} or {term2}?"
            ])
            response += f"\n\n{followup}"
            
        return response

    def create_followup_comparison_response(self, term1: str, term2: str) -> str:
        """Create a follow-up response for comparison questions."""
        response_templates = [
            f"Both {term1} and {term2} are important concepts in finance, but their relative importance depends on your specific situation. "
            f"{term1} might be more relevant if you're dealing with {random.choice(['short-term planning', 'operational decisions', 'specific investments', 'risk assessment'])}, "
            f"while {term2} could be more important when considering {random.choice(['long-term strategy', 'broader financial planning', 'market analysis', 'portfolio diversification'])}. "
            f"Understanding both gives you a more comprehensive financial perspective, allowing you to make better-informed decisions across various scenarios.",
            
            f"Professionally, investors use {term1} primarily for {random.choice(['analyzing market trends', 'evaluating investment opportunities', 'assessing risk profiles', 'making strategic decisions'])}, "
            f"whereas {term2} is often utilized for {random.choice(['portfolio optimization', 'performance benchmarking', 'financial forecasting', 'compliance considerations'])}. "
            f"The concepts are complementary rather than competing, and proficiency in both enhances one's ability to navigate complex financial landscapes effectively.",
            
            f"{term1} is typically more relevant in situations involving {random.choice(['active trading', 'short-term investments', 'market volatility', 'specific financial instruments'])}, "
            f"while {term2} comes into play more when dealing with {random.choice(['long-term planning', 'broader economic trends', 'fundamental analysis', 'strategic asset allocation'])}. "
            f"However, they often intersect, and understanding their relationship can provide valuable insights for comprehensive financial decision-making.",
            
            f"Several related concepts connect both {term1} and {term2}, including {random.choice(['risk management', 'asset valuation', 'market efficiency', 'economic indicators'])} "
            f"and {random.choice(['portfolio theory', 'financial regulations', 'capital structure', 'liquidity considerations'])}. "
            f"These form part of the broader financial ecosystem where different concepts interact and influence each other, creating a rich tapestry of knowledge that informs effective financial practice."
        ]
        
        return random.choice(response_templates)

    def create_qa_pair(self, definition: Dict[str, str], with_conversation: bool = False) -> Dict[str, Any]:
        """Create a QA pair from a definition."""
        term = definition["term"]
        definition_text = definition["definition"]
        
        # Generate a question from the template - 50% chance to use conversational template
        if random.random() < 0.5:
            question_templates = self.conversational_q_templates
        else:
            question_templates = self.definition_q_templates
            
        question = random.choice(question_templates).format(term=term)
        
        # Enhance the definition
        answer = self.enhance_definition(term, definition_text)
        
        # Create unique conversation ID
        conv_id = hashlib.sha256(f"{term}_{question}".encode()).hexdigest()
        
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
                    "content": answer
                }
            ]
        # Extended conversation format
        else:
            # 70% chance to use intro_generator for more natural conversations (increased from 50%)
            if random.random() < 0.7:
                messages = self.intro_generator.generate_conversation()
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
            else:
                # Use our standard conversation starters
                starter = random.choice(self.conversation_starters)
                
                messages = [
                    {
                        "role": "system",
                        "content": random.choice(self.system_prompts)
                    },
                    {
                        "role": "user",
                        "content": starter["user"]
                    },
                    {
                        "role": "assistant",
                        "content": starter["assistant"]
                    },
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            
            # Generate 2-4 follow-up exchanges (increased from just one)
            num_followups = random.randint(2, 4)
            
            for _ in range(num_followups):
                # Additional follow-up questions
                deeper_followups = [
                    f"Could you give me a specific example of {term} in action?",
                    f"What are some common misconceptions about {term}?",
                    f"How does {term} affect investment decisions?",
                    f"What should beginners know about {term}?",
                    f"Are there any risks associated with {term}?",
                    f"How has the concept of {term} evolved over time?",
                    f"What's the significance of {term} in today's market?",
                    f"How do professionals use {term} in their analysis?",
                    f"Can {term} impact personal financial planning?",
                    f"What's the relationship between {term} and market performance?",
                    f"How would changes in {term} affect investment strategies?",
                    f"What factors influence {term}?",
                    f"How is {term} different in various market conditions?",
                    f"What tools or metrics are used to analyze {term}?",
                    f"How does {term} relate to risk management?"
                ]

                conversational_deeper_followups = [
                    f"I'm curious about real-world applications of {term}. Can you share some examples?",
                    f"A friend told me something about {term} that seems wrong. What are common misunderstandings?",
                    f"I'm new to investing - how would {term} affect my investment choices?",
                    f"This is fascinating! What else should I know about {term} as a beginner?",
                    f"Should I be worried about any risks related to {term}?",
                    f"How has {term} changed with modern financial markets?",
                    f"Why is everyone talking about {term} lately?",
                    f"I've heard professionals mention {term} - how do they use it?",
                    f"Could {term} affect my personal financial planning?",
                    f"I've noticed {term} mentioned in market news. What's the connection?",
                    f"If {term} changes, should I adjust my investment strategy?",
                    f"What tends to cause changes in {term}?",
                    f"Does {term} work differently in bull vs bear markets?",
                    f"How do analysts measure or evaluate {term}?",
                    f"Should I consider {term} when thinking about risk?"
                ]

                # 60% chance to use conversational follow-up
                if random.random() < 0.6:
                    user_followup = random.choice(conversational_deeper_followups)
                else:
                    user_followup = random.choice(deeper_followups)

                # Generate detailed follow-up response
                assistant_followup = self.create_detailed_followup_response(term, user_followup)
                
                messages.append({"role": "user", "content": user_followup})
                messages.append({"role": "assistant", "content": assistant_followup})

        return {
            "messages": messages,
            "metadata": {
                "source": "financial_definitions",
                "term": term,
                "conversation_id": conv_id,
                "type": "definition",
                "turns": len(messages) // 2  # Track conversation length
            }
        }

    def create_detailed_followup_response(self, term: str, question: str) -> str:
        """Create detailed follow-up responses based on question type."""
        
        # Example-related responses templates and choices
        example_scenarios = [
            'investors evaluate company performance', 
            'analysts review market trends', 
            'traders make decisions', 
            'financial planners work with clients'
        ]
        example_uses = [
            'assess risk levels', 
            'measure potential returns', 
            'evaluate market conditions', 
            'make informed decisions'
        ]
        example_contexts = [
            'during market volatility', 
            'in a bull market', 
            'when interest rates change', 
            'during economic uncertainty'
        ]
        example_benefits = [
            'identify opportunities', 
            'minimize risks', 
            'optimize strategies', 
            'maximize returns'
        ]
        
        # More example choices
        planning_scenarios = [
            "you're planning your retirement", 
            "you're starting to invest", 
            "you're analyzing a company", 
            "you're building a portfolio"
        ]
        planning_benefits = [
            'make better decisions', 
            'avoid common pitfalls', 
            'optimize your strategy', 
            'achieve better results'
        ]
        planning_methods = [
            'providing clear metrics', 
            'offering valuable insights', 
            'highlighting important factors', 
            'identifying key trends'
        ]
        
        # Professional scenario choices
        professional_roles = [
            'A financial advisor is helping a client', 
            'An investor is analyzing markets', 
            'A trader is making decisions', 
            'A portfolio manager is rebalancing accounts'
        ]
        professional_uses = [
            'evaluate options', 
            'assess conditions', 
            'measure performance', 
            'guide decisions'
        ]
        professional_outcomes = [
            'achieve better outcomes', 
            'minimize potential risks', 
            'optimize their strategy', 
            'make informed choices'
        ]
        
        # Example-related responses using the separated choices
        example_responses = [
            f"Here's a practical example of {term} in action: When {random.choice(example_scenarios)}, they often look at {term} to {random.choice(example_uses)}. For instance, {random.choice(example_contexts)}, {term} becomes particularly important because it helps {random.choice(example_benefits)}.",
            
            f"Let me break this down with a real-world scenario. Imagine {random.choice(planning_scenarios)}. Understanding {term} would help you {random.choice(planning_benefits)} by {random.choice(planning_methods)}.",
            
            f"Consider this scenario: {random.choice(professional_roles)}. They would use {term} to {random.choice(professional_uses)}. This helps them {random.choice(professional_outcomes)}."
        ]

        # Risk-related responses choices
        risk_factors = [
            'market conditions can affect', 
            'economic changes might impact', 
            'regulatory changes could influence', 
            'global events may alter'
        ]
        risk_mistakes = [
            'misunderstanding its implications', 
            'overlooking important factors', 
            'not monitoring changes', 
            'making assumptions'
        ]
        risk_consequences = [
            'suboptimal decisions', 
            'missed opportunities', 
            'unexpected losses', 
            'strategic errors'
        ]
        
        # Main risks choices
        risk_type1 = [
            'market volatility', 
            'timing considerations', 
            'measurement accuracy', 
            'interpretation challenges'
        ]
        risk_type2 = [
            'implementation difficulties', 
            'regulatory changes', 
            'economic impacts', 
            'structural shifts'
        ]
        risk_type3 = [
            'behavioral biases', 
            'technical limitations', 
            'practical constraints', 
            'systemic factors'
        ]
        risk_management = [
            'monitor these risks', 
            'understand their impact', 
            'have mitigation strategies', 
            'stay informed'
        ]
        
        # Understanding risks choices
        consideration_factors = [
            'market conditions', 
            'economic factors', 
            'timing aspects', 
            'implementation challenges'
        ]
        awareness_factors = [
            'potential limitations', 
            'common pitfalls', 
            'practical constraints', 
            'external factors'
        ]
        benefit_factors = [
            'making better decisions', 
            'managing expectations', 
            'optimizing outcomes', 
            'reducing uncertainty'
        ]
        
        # Risk-related responses using separated choices
        risk_responses = [
            f"When it comes to risks associated with {term}, there are several key considerations. First, {random.choice(risk_factors)} how {term} behaves. Additionally, {random.choice(risk_mistakes)} can lead to {random.choice(risk_consequences)}.",
            
            f"The main risks related to {term} include: 1) {random.choice(risk_type1)}, 2) {random.choice(risk_type2)}, and 3) {random.choice(risk_type3)}. It's important to {random.choice(risk_management)}.",
            
            f"Understanding the risks of {term} is crucial. You need to consider {random.choice(consideration_factors)}, while also being aware of {random.choice(awareness_factors)}. This helps in {random.choice(benefit_factors)}."
        ]

        # Market impact choices
        market_roles = [
            'influences investment decisions', 
            'affects market behavior', 
            'impacts trading strategies', 
            'shapes financial planning'
        ]
        market_relevance = [
            'changing economic factors', 
            'evolving investor preferences', 
            'technological advances', 
            'regulatory developments'
        ]
        
        # Relationship choices
        relationship_factors = [
            'multiple factors', 
            'various indicators', 
            'different timeframes', 
            'diverse influences'
        ]
        relationship_impacts = [
            'affect outcomes', 
            'impact results', 
            'shape trends', 
            'drive changes'
        ]
        relationship_benefits = [
            'making better decisions', 
            'optimizing strategies', 
            'managing risks', 
            'improving results'
        ]
        
        # Market condition choices
        condition_types = [
            'periods of volatility', 
            'economic uncertainty', 
            'market transitions', 
            'rapid changes'
        ]
        condition_actions = [
            'monitor developments', 
            'adjust strategies', 
            'reassess positions', 
            'update approaches'
        ]

        # Market impact responses using separated choices
        market_responses = [
            f"In today's market, {term} plays a crucial role because it {random.choice(market_roles)}. Current market conditions make it particularly relevant due to {random.choice(market_relevance)}.",
            
            f"The relationship between {term} and market performance is complex. It involves {random.choice(relationship_factors)} that can {random.choice(relationship_impacts)}. Understanding this relationship helps in {random.choice(relationship_benefits)}.",
            
            f"Market conditions can significantly affect how {term} behaves. During {random.choice(condition_types)}, it becomes especially important to {random.choice(condition_actions)} related to {term}."
        ]

        # Choose response type based on question content
        if any(word in question.lower() for word in ['example', 'instance', 'case']):
            response = random.choice(example_responses)
        elif any(word in question.lower() for word in ['risk', 'danger', 'concern']):
            response = random.choice(risk_responses)
        elif any(word in question.lower() for word in ['market', 'trend', 'performance']):
            response = random.choice(market_responses)
        else:
            # Default to a combination response
            response = random.choice(example_responses + risk_responses + market_responses)

        # 40% chance to add a follow-up question
        if random.random() < 0.4:
            followup = random.choice([
                f"Would you like me to elaborate on any specific aspect of {term}?",
                f"Is there a particular part of this explanation you'd like me to clarify?",
                f"How does this information relate to your financial goals?",
                f"Would you like to explore any related concepts?",
                f"Does this help explain the role of {term} in your financial planning?"
            ])
            response += f"\n\n{followup}"

        return response

    def generate_dataset(self) -> None:
        """Generate the dataset with definition QA pairs."""
        if not self.definitions:
            self.load_definitions()
            
        # Calculate number of samples for each type
        num_comparisons = int(self.num_samples * self.comparison_percentage)
        num_remaining = self.num_samples - num_comparisons
        num_with_conversation = int(num_remaining * 0.6)  # 40% with extended conversation
        num_simple = num_remaining - num_with_conversation  # 60% with simple Q&A
        
        logger.info(f"Generating {num_simple} simple Q&A pairs")
        
        # Generate samples with replacement if we need more samples than definitions
        for _ in tqdm(range(num_simple), desc="Generating simple Q&A pairs"):
            definition = random.choice(self.definitions)
            qa_pair = self.create_qa_pair(definition, with_conversation=False)
            self.dataset.append(qa_pair)
            
        logger.info(f"Generating {num_with_conversation} conversation-style Q&A pairs")
        
        for _ in tqdm(range(num_with_conversation), desc="Generating conversation Q&A pairs"):
            definition = random.choice(self.definitions)
            qa_pair = self.create_qa_pair(definition, with_conversation=True)
            self.dataset.append(qa_pair)
            
        logger.info(f"Generating {num_comparisons} comparison Q&A pairs")
        
        for _ in tqdm(range(num_comparisons), desc="Generating comparison Q&A pairs"):
            # Pick two different random terms
            term1_def = random.choice(self.definitions)
            term2_def = random.choice(self.definitions)
            
            # Ensure we have two different terms
            attempts = 0
            while term1_def["term"] == term2_def["term"] and attempts < 10:
                term2_def = random.choice(self.definitions)
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
        self.load_definitions()
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
