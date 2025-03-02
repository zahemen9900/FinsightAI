
import random
import logging
import argparse
from pathlib import Path
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

class FinanceQuestionGenerator:
    """Generate finance-related questions for manual response creation."""
    
    def __init__(self, output_file: str = "finance_questions.txt", num_questions: int = 100):
        self.output_file = Path(output_file)
        self.num_questions = num_questions
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Company-related question templates
        self.company_question_templates = [
            "What was {company}'s revenue growth in their latest quarter?",
            "How did {company} perform compared to analyst expectations?",
            "What are {company}'s main products or services?",
            "How does {company} generate most of its revenue?",
            "What is {company}'s market position in their industry?",
            "How has {company}'s strategy evolved over the past year?",
            "What competitive advantages does {company} have?",
            "What risks does {company} face in their business?",
            "Has {company} announced any major initiatives recently?",
            "What is {company}'s approach to sustainability and ESG?",
            "How does {company}'s valuation compare to its peers?",
            "What are analysts saying about {company}'s stock?",
            "How might rising interest rates affect {company}?",
            "What are {company}'s growth prospects for the next few years?",
            "How diversified is {company}'s revenue stream?",
            "What is {company}'s market share in its primary business?",
            "How does {company}'s management communicate with investors?",
            "What are the key metrics I should look at to evaluate {company}?",
            "How has {company}'s stock performed over the past year?",
            "What recent acquisitions has {company} made?",
            "How is {company} affected by current macroeconomic conditions?",
            "What are the biggest challenges facing {company} right now?",
            "How does {company}'s profit margin compare to industry standards?",
            "Is {company} considered a leader in innovation within its sector?",
            "What is {company}'s dividend policy?"
        ]
        
        # Investment strategy questions
        self.investment_question_templates = [
            "How should I adjust my portfolio for the current economic environment?",
            "What asset allocation would you recommend for someone in their 30s?",
            "How much international exposure should I have in my investment portfolio?",
            "What's the optimal balance between stocks and bonds for a retiree?",
            "Should I be concerned about inflation when planning my investments?",
            "How can I build a dividend-focused portfolio?",
            "What are the benefits and drawbacks of index investing?",
            "How do I evaluate if a stock is overvalued or undervalued?",
            "What metrics should I look at when analyzing a potential investment?",
            "How can I incorporate ESG factors into my investment decisions?",
            "What's a reasonable expected rate of return for a diversified portfolio?",
            "How often should I rebalance my investment portfolio?",
            "What tax-efficient investing strategies would you recommend?",
            "How can I protect my portfolio against market downturns?",
            "What alternative investments should I consider beyond stocks and bonds?",
            "How can I evaluate a company's competitive advantages before investing?",
            "What's the best approach to dollar-cost averaging?",
            "How should I think about risk tolerance when investing?",
            "What's your view on active versus passive investment strategies?",
            "How much cash should I keep in my investment portfolio?",
            "How do interest rates affect different types of investments?",
            "What sectors do you think will outperform in the next few years?",
            "How can I build a portfolio focused on generating passive income?",
            "Should I consider investing in REITs as part of my portfolio?",
            "How do I evaluate a mutual fund's performance and expenses?"
        ]
        
        # Financial planning questions
        self.planning_question_templates = [
            "How much should I save for retirement if I'm starting at age 35?",
            "What's the optimal strategy for paying down multiple debts?",
            "How do I calculate how much life insurance I need?",
            "What percentage of my income should go toward housing costs?",
            "How can I optimize my tax situation when saving for retirement?",
            "What strategies can help me save for my children's education?",
            "How do I balance saving for retirement vs. paying off my mortgage?",
            "What should I consider when deciding between a Roth IRA and Traditional IRA?",
            "How can I determine if I'm on track to meet my retirement goals?",
            "What's the best approach to estate planning for a young family?",
            "How much should I have in my emergency fund?",
            "What financial steps should I take before buying a home?",
            "How do I create a sustainable retirement withdrawal strategy?",
            "What should I consider when choosing between pension options?",
            "How can I minimize taxes during retirement?",
            "What's the optimal strategy for claiming Social Security benefits?",
            "How do I account for healthcare costs in retirement planning?",
            "What's the most tax-efficient way to make charitable donations?",
            "How should I adjust my financial plan during high inflation periods?",
            "What financial preparations should I make before having a child?",
            "How can I protect my assets from potential lawsuits or creditors?",
            "What's the most efficient way to transfer wealth to the next generation?",
            "How do I create a budget that I'll actually stick to?",
            "What financial metrics should I track to ensure I'm meeting my goals?",
            "How should I prioritize different financial goals like retirement, education, and home ownership?"
        ]
        
        # Market analysis questions
        self.market_question_templates = [
            "What factors are currently driving the stock market?",
            "How do you interpret the current yield curve?",
            "What impact might the Federal Reserve's policies have on the markets?",
            "How could current geopolitical tensions affect global markets?",
            "What leading economic indicators should I pay attention to right now?",
            "How might inflation trends impact different market sectors?",
            "What does the current market volatility tell us about investor sentiment?",
            "How do corporate earnings reports influence overall market direction?",
            "What technical indicators suggest where the market might be heading?",
            "How do you evaluate whether we're in a bull or bear market?",
            "What sectors tend to perform best during economic slowdowns?",
            "How do currency fluctuations impact multinational companies?",
            "What's the relationship between consumer confidence and market performance?",
            "How do supply chain issues affect different market sectors?",
            "What impact do government deficits have on long-term market returns?",
            "How do commodities markets interact with equity markets?",
            "What effect does an aging population have on economic growth prospects?",
            "How do employment reports influence market performance?",
            "What market signals might indicate a recession is approaching?",
            "How do housing market trends affect the broader economy?",
            "What's the historical correlation between interest rates and stock market performance?",
            "How do institutional investors currently view market valuations?",
            "What role does market liquidity play in price movements?",
            "How do international markets influence U.S. market performance?",
            "What current market anomalies might present investment opportunities?"
        ]
        
        # Financial product questions
        self.product_question_templates = [
            "What's the difference between ETFs and mutual funds?",
            "How do I choose the right type of life insurance?",
            "What should I look for in a high-yield savings account?",
            "How do municipal bonds work and what are their tax advantages?",
            "What are the pros and cons of adjustable-rate vs. fixed-rate mortgages?",
            "How do target-date retirement funds work?",
            "What should I consider when evaluating annuities?",
            "How do TIPS (Treasury Inflation-Protected Securities) protect against inflation?",
            "What are the advantages of HSAs (Health Savings Accounts) for long-term planning?",
            "How do I evaluate the fees in my 401(k) plan options?",
            "What's the difference between term and whole life insurance?",
            "How do closed-end funds differ from open-end mutual funds?",
            "What are the risks associated with high-yield corporate bonds?",
            "How do options contracts work and how are they priced?",
            "What are the benefits of dollar-cost averaging with index funds?",
            "How do reverse mortgages work and when might they be appropriate?",
            "What are the different types of REITs and how do they generate returns?",
            "How do I evaluate the credit quality of a bond?",
            "What are structured notes and how do they work?",
            "How do margin accounts work and what are the risks?",
            "What are the differences between various cryptocurrency investments?",
            "How do robo-advisors compare to traditional financial advisors?",
            "What are the advantages of factor-based ETFs?",
            "How do I determine if a variable annuity is a good investment?",
            "What are the key features to look for in a rewards credit card?"
        ]
        
        # Advanced financial concept questions
        self.concept_question_templates = [
            "Can you explain modern portfolio theory in simple terms?",
            "How does the capital asset pricing model work?",
            "What is the efficient market hypothesis and what are its limitations?",
            "How do you calculate the weighted average cost of capital (WACC)?",
            "What's the difference between systematic and unsystematic risk?",
            "Can you explain the concept of duration in bond investing?",
            "How does arbitrage pricing theory differ from CAPM?",
            "What's the significance of the Sharpe ratio when evaluating investments?",
            "How do you interpret a company's price-to-earnings ratio?",
            "What is quantitative easing and how does it affect markets?",
            "Can you explain how derivatives are used for hedging?",
            "What's the role of the yield curve in economic forecasting?",
            "How do central banks influence interest rates and money supply?",
            "What is financial leverage and how does it amplify returns and risks?",
            "How do market makers provide liquidity to markets?",
            "What is the Black-Scholes model used for?",
            "How does behavioral finance challenge traditional economic theories?",
            "What is the difference between fiscal and monetary policy?",
            "How do credit default swaps function as insurance against defaults?",
            "What is the equity risk premium and why is it important?",
            "How do contango and backwardation affect commodity futures pricing?",
            "What is the significance of Tobin's Q ratio in valuation?",
            "How does the Fed's dot plot influence market expectations?",
            "What is the relationship between inflation and unemployment according to the Phillips Curve?",
            "How do currency carry trades exploit interest rate differentials?"
        ]

        # Popular companies for examples
        self.companies = [
            "Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla", "Nvidia", 
            "JPMorgan Chase", "Bank of America", "Wells Fargo", "Citigroup", "Goldman Sachs",
            "Walmart", "Target", "Costco", "Home Depot", "Lowe's", "Nike", "Starbucks",
            "McDonald's", "Coca-Cola", "PepsiCo", "Johnson & Johnson", "Pfizer", "Merck",
            "Exxon Mobil", "Chevron", "Shell", "BP", "ConocoPhillips", "Disney", "Netflix",
            "Comcast", "AT&T", "Verizon", "T-Mobile", "Intel", "AMD", "Qualcomm", "Cisco",
            "Oracle", "Salesforce", "Adobe", "IBM", "Dell", "HP", "Boeing", "Lockheed Martin",
            "General Electric", "3M", "Caterpillar", "Deere", "Ford", "General Motors"
        ]
        
        # Comparison question templates
        self.comparison_question_templates = [
            "How do {company1} and {company2} compare in terms of financial performance?",
            "What are the key differences between {company1} and {company2}'s business models?",
            "How does {company1}'s market share compare to {company2}'s in their industry?",
            "Which company has better growth prospects: {company1} or {company2}?",
            "How do the dividend policies of {company1} and {company2} differ?",
            "What are the differences in management style between {company1} and {company2}?",
            "How does {company1}'s debt load compare to {company2}'s?",
            "Which company is more innovative: {company1} or {company2}?",
            "How do {company1} and {company2} differ in their approach to sustainability?",
            "What are the relative strengths and weaknesses of {company1} versus {company2}?",
            "How do analysts' ratings compare between {company1} and {company2}?",
            "Which stock has performed better over the past 5 years: {company1} or {company2}?",
            "How do the profit margins of {company1} and {company2} compare?",
            "Which company has better international growth: {company1} or {company2}?",
            "How do {company1} and {company2} differ in their capital allocation strategies?"
        ]

    def generate_company_question(self) -> str:
        """Generate a question about a specific company."""
        company = random.choice(self.companies)
        template = random.choice(self.company_question_templates)
        return template.format(company=company)

    def generate_comparison_question(self) -> str:
        """Generate a question comparing two companies."""
        company1, company2 = random.sample(self.companies, 2)
        template = random.choice(self.comparison_question_templates)
        return template.format(company1=company1, company2=company2)

    def generate_investment_question(self) -> str:
        """Generate a question about investment strategies."""
        return random.choice(self.investment_question_templates)

    def generate_planning_question(self) -> str:
        """Generate a question about financial planning."""
        return random.choice(self.planning_question_templates)

    def generate_market_question(self) -> str:
        """Generate a question about market analysis."""
        return random.choice(self.market_question_templates)

    def generate_product_question(self) -> str:
        """Generate a question about financial products."""
        return random.choice(self.product_question_templates)

    def generate_concept_question(self) -> str:
        """Generate a question about advanced financial concepts."""
        return random.choice(self.concept_question_templates)

    def create_conversation_block(self, question: str) -> str:
        """Create a conversation block with a question and blank assistant response."""
        return f"User: {question}\nAssistant: [leave-blank]\n\n"

    def generate_questions(self) -> List[str]:
        """Generate a variety of finance-related questions."""
        question_types = [
            self.generate_company_question,
            self.generate_comparison_question,
            self.generate_investment_question,
            self.generate_planning_question,
            self.generate_market_question,
            self.generate_product_question,
            self.generate_concept_question
        ]
        
        # Weights for different question types
        weights = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]
        
        # Generate questions based on weights
        questions = []
        for _ in range(self.num_questions):
            question_gen_func = random.choices(question_types, weights=weights, k=1)[0]
            question = question_gen_func()
            questions.append(question)
            
        return questions

    def write_questions_to_file(self, questions: List[str]) -> None:
        """Write generated questions to a text file in the specified format."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for question in questions:
                conversation_block = self.create_conversation_block(question)
                f.write(conversation_block)
                
        logger.info(f"Successfully wrote {len(questions)} questions to {self.output_file}")

    def run(self) -> None:
        """Execute the full question generation process."""
        logger.info(f"Starting finance question generation: {self.num_questions} questions")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            gen_task = progress.add_task("[green]Generating questions...", total=1)
            
            # Generate questions
            questions = self.generate_questions()
            progress.update(gen_task, advance=0.5)
            
            # Write questions to file
            self.write_questions_to_file(questions)
            progress.update(gen_task, advance=0.5)
        
        logger.info(f"Generated question categories:")
        logger.info(f"  Company-specific: ~{int(self.num_questions * 0.2)}")
        logger.info(f"  Company comparisons: ~{int(self.num_questions * 0.15)}")
        logger.info(f"  Investment strategies: ~{int(self.num_questions * 0.15)}")
        logger.info(f"  Financial planning: ~{int(self.num_questions * 0.15)}")
        logger.info(f"  Market analysis: ~{int(self.num_questions * 0.15)}")
        logger.info(f"  Financial products: ~{int(self.num_questions * 0.1)}")
        logger.info(f"  Advanced concepts: ~{int(self.num_questions * 0.1)}")
        logger.info(f"Output file: {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate finance questions for manual response creation")
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="/home/zahemen/datasets/finance_questions.txt",
        help="Output file for generated questions"
    )
    
    parser.add_argument(
        "--count", 
        type=int, 
        default=500,
        help="Number of questions to generate"
    )
    
    args = parser.parse_args()
    
    generator = FinanceQuestionGenerator(args.output, args.count)
    generator.run()

if __name__ == "__main__":
    main()
