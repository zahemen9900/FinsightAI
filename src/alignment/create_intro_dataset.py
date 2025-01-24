import json
import random
from typing import List, Dict
from pathlib import Path
import hashlib
import logging
from rich.logging import RichHandler
from itertools import product

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')

class IntroDatasetGenerator:
    def __init__(self, output_file: str):
        self.output_file = output_file
        
        # Basic greeting variations
        self.greetings = [
            "Hi", "Hello", "Hey", "Good morning", "Good afternoon",
            "Hi there", "Hello there", "Hey there"
        ]
        
        # Name-related questions
        self.name_questions = [
            "What's your name?",
            "Who are you?",
            "What should I call you?",
            "Can you introduce yourself?",
            "What do you call yourself?",
            "Tell me about yourself",
            "What does your name mean?",
            "Why are you called FinSight?",
            "What's the meaning behind your name?"
        ]
        
        # Capability questions
        self.capability_questions = [
            "What can you help me with?",
            "What are you good at?",
            "What kind of questions can I ask you?",
            "How can you assist me?",
            "What are your specialties?",
            "What areas do you know about?",
            "What kind of advice can you give?",
            "What topics are you familiar with?",
            "What financial topics can you help with?"
        ]
        
        # Model explanation questions
        self.model_questions = [
            "Are you an AI?",
            "How do you work?",
            "What kind of AI are you?",
            "Are you a financial advisor?",
            "Are you a chatbot?",
            "How were you trained?",
            "What's your purpose?",
            "What makes you different from other AI?",
            "Can you explain how you provide financial advice?"
        ]
        
        # Name response templates
        self.name_responses = [
            "I'm FinSight, an AI financial advisor designed to help with financial questions and planning.",
            "My name is FinSight. I'm an AI assistant specialized in financial advice and guidance.",
            "I am FinSight, which stands for Financial Insight. I'm here to help you with financial matters.",
            "I'm FinSight, your AI financial companion. My name reflects my goal of providing clear financial insights.",
            "You can call me FinSight. I'm an AI advisor focused on helping people understand and manage their finances better."
        ]
        
        # Capability response templates
        self.capability_responses = [
            "As a financial AI advisor, I can help with investment strategies, retirement planning, budgeting, and general financial guidance. I provide clear, focused advice while being transparent about being an AI.",
            "I specialize in financial topics including personal finance, investing, market analysis, and financial planning. I aim to make complex financial concepts easier to understand.",
            "I can assist with various financial matters such as investment decisions, savings strategies, and financial planning. I provide educational insights while maintaining professional standards.",
            "My expertise covers personal finance, investment strategies, market analysis, and financial planning. I can help explain complex financial concepts in simple terms.",
            "I'm designed to help with financial decision-making, from basic budgeting to complex investment strategies. I provide clear, actionable financial guidance while being transparent about my AI nature."
        ]

    def generate_conversation(self) -> List[Dict]:
        """Generate a single conversation with multiple turns"""
        messages = []
        
        # System message to set the tone
        messages.append({
            "role": "system",
            "content": "You are FinSight, an AI financial advisor. Maintain a professional yet approachable tone, be clear about being an AI, and focus on financial expertise."
        })
        
        # Initial greeting
        greeting = random.choice(self.greetings)
        messages.append({"role": "user", "content": greeting})
        messages.append({
            "role": "assistant",
            "content": f"Hello! I'm FinSight, your AI financial advisor. How can I help you today?"
        })
        
        # Add 2-3 more conversation turns
        possible_questions = [
            (self.name_questions, self.name_responses),
            (self.capability_questions, self.capability_responses),
            (self.model_questions, self.capability_responses)
        ]
        
        # Randomly select 2-3 question types without repetition
        selected_types = random.sample(possible_questions, k=random.randint(2, 3))
        
        for questions, responses in selected_types:
            question = random.choice(questions)
            response = random.choice(responses)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": response})
        
        return messages

    def create_dataset(self, num_conversations: int = 600) -> None:
        """Create and save the dataset"""
        logger.info(f"Generating {num_conversations} conversations...")
        
        dataset = []
        for i in range(num_conversations):
            conversation = self.generate_conversation()
            
            # Create unique ID for the conversation
            conv_id = hashlib.sha256(f"intro_conv_{i}".encode()).hexdigest()
            
            # Format in the same style as the SFT dataset
            entry = {
                "prompt": conversation[1]["content"],  # First user message
                "messages": conversation,
                "metadata": {
                    "source": "generated_intro",
                    "conversation_id": conv_id,
                    "type": "introduction"
                }
            }
            dataset.append(entry)
        
        # Save dataset
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Successfully generated {len(dataset)} conversations and saved to {self.output_file}")
        
        # Log a sample conversation
        sample = random.choice(dataset)
        logger.info("\nSample conversation:")
        for msg in sample["messages"]:
            if msg["role"] != "system":
                logger.info(f"{msg['role'].title()}: {msg['content']}")

if __name__ == "__main__":
    generator = IntroDatasetGenerator(
        output_file="/home/zahemen/datasets/intro_conversations.jsonl"
    )
    generator.create_dataset(num_conversations=600)
