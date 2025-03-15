import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from dataclasses import dataclass
from decimal import Decimal
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

@dataclass
class FinancialProfile:
    """Represents a financial profile for generating personalized scenarios."""
    income: int
    age: int
    savings: int
    debt: Dict[str, int]
    dependents: int
    risk_tolerance: str
    investment_horizon: int
    tax_bracket: int
    goals: List[str]

class AdvancedFinanceQuestionGenerator:
    """Generate advanced finance-related questions with detailed scenarios."""
    
    def __init__(self, output_file: str = "advanced_finance_questions.txt", num_questions: int = 1000):
        self.output_file = Path(output_file)
        self.num_questions = num_questions
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize templates
        self._init_templates()
        self._init_financial_data()
    
    def _init_financial_data(self):
        """Initialize financial data for generating realistic scenarios."""
        self.income_brackets = [
            (30000, 50000), (50001, 75000), (75001, 100000),
            (100001, 150000), (150001, 250000), (250001, 500000)
        ]
        
        self.debt_types = {
            "student_loans": (10000, 100000),
            "mortgage": (100000, 800000),
            "car_loan": (5000, 50000),
            "credit_card": (1000, 30000),
            "personal_loan": (5000, 50000)
        }
        
        self.risk_profiles = ["Conservative", "Moderate", "Aggressive"]
        
        self.investment_goals = [
            "Retirement", "Home Purchase", "Children's Education",
            "Starting a Business", "Building Wealth", "Passive Income"
        ]
        
        self.asset_classes = [
            "Stocks", "Bonds", "Real Estate", "Commodities",
            "Cryptocurrencies", "Cash Equivalents", "Alternative Investments"
        ]
        
        self.market_conditions = [
            "Bull Market", "Bear Market", "High Inflation",
            "Low Interest Rates", "Economic Recession", "Market Volatility"
        ]
        
        self.tax_considerations = [
            "Capital Gains", "Dividend Income", "Tax-Loss Harvesting",
            "Retirement Account Distributions", "Estate Tax Planning"
        ]

    def _init_templates(self):
        """Initialize question templates for each category."""
        # Personalized Financial Advice Templates
        self.personalized_advice_templates = [
            # Basic Financial Planning
            "I make ${income} per month. How should I budget this for essential expenses, savings, and investments?",
            "I have ${amount} saved up. What's the best way to start investing with this amount?",
            "Making ${income} annually, how can I create an emergency fund while paying off ${debt_amount} in {debt_type}?",
            "I'm {age} with ${savings} saved. How should I plan for retirement?",
            
            # Intermediate Planning
            "With an annual income of ${income} and ${savings} in savings, how should I allocate my monthly budget "
            "to reach my goal of {goal} in {timeframe} years?",
            "I'm {age} years old, earning ${income}/year with ${debt_amount} in {debt_type} debt. "
            "What's the most efficient strategy to become debt-free while still saving for retirement?",
            "How should I restructure my budget after a salary increase from ${old_income} to ${new_income}?",
            
            # Complex Planning
            "With ${income} income, ${savings} savings, and ${debt_amount} total debt across {debt_types}, "
            "how should I optimize my financial strategy for {goals}?",
            "I'm {age} with {dependents} dependents, ${income} income, and a {risk_tolerance} risk tolerance. "
            "How should I balance college savings, retirement, and ${debt_amount} mortgage debt?",
            "Given my {risk_tolerance} risk profile and ${portfolio_size} portfolio, how should I rebalance "
            "between {asset_types} considering my {timeframe}-year horizon and {goal}?",
            
            # Career-based Financial Planning
            "As a {age}-year-old professional considering a career change from a ${income} position to entrepreneurship, "
            "how should I adjust my finances with ${savings} in savings and {dependents} dependents?",
            "I'm transitioning to a contract role paying ${income} from a salaried position. How should I manage taxes, "
            "health insurance, and retirement with ${savings} saved and a {risk_tolerance} risk profile?",
            
            # Life Event Financial Planning
            "I'm planning to buy a home in {timeframe} years with ${savings} saved. How should I prepare financially "
            "while earning ${income} and managing ${debt_amount} in student loans?",
            "I'm expecting a child in 6 months. With ${income} household income and ${savings} savings, how should "
            "we adjust our financial plan, considering our {goal} and {timeframe}-year timeline?",
            "I'm {age} years old and recently inherited ${savings}. How should I incorporate this into my financial plan "
            "with my existing ${income} income and {risk_tolerance} risk tolerance?",
            
            # Geographic Financial Planning
            "I'm relocating from a low-cost area to a high-cost city for a job paying ${income}. How should I adjust my "
            "budget and investment strategy with ${savings} savings and ${debt_amount} remaining mortgage?",
            "I'm planning to retire abroad in {timeframe} years. How should I restructure my ${portfolio_size} portfolio "
            "and plan for currency exchange risks with {risk_tolerance} risk tolerance?"
        ]

        # Investment Strategy Templates
        self.investment_strategy_templates = [
            # Basic Investment
            "What's the best way to invest ${amount} as a beginner?",
            "Should I invest in individual stocks or stick to index funds with ${amount}?",
            "How do I start building a dividend portfolio with ${amount}?",
            "I've only got ${amount} to invest after paying bills. Is it even worth investing such a small amount?",
            "I've never invested before and have ${amount} saved up. What's the safest way to start?",
            
            # Intermediate Investment
            "How should I rebalance my portfolio of {assets} during {market_condition}?",
            "What's the optimal asset allocation for a ${portfolio_size} portfolio with a {timeframe}-year horizon?",
            "How should I diversify ${amount} across {num_assets} different asset classes?",
            "I've been putting ${amount} monthly into my 401(k) but the balance seems stagnant. What am I doing wrong?",
            "Between paying off my ${debt_amount} debt and investing ${amount}, which should I prioritize?",
            
            # Advanced Investment
            "With a ${portfolio_size} portfolio focused on {strategy}, what adjustments would you recommend "
            "given {market_condition} and {economic_indicators}?",
            "How should I implement a factor investing strategy with ${amount} focusing on {factors}?",
            "What options strategies would be suitable for a ${portfolio_size} portfolio during {market_condition}?",
            
            # Financial Hardship Scenarios
            "I lost my job and only have ${amount} in savings. Should I keep it as cash or invest some portion?",
            "After medical bills, I only have ${amount} left each month. Is investing even possible for me?",
            "I'm working two jobs and can save ${amount} monthly. What's the best investment strategy for someone with limited time?",
            "I'm behind on retirement at age {age} with only ${savings} saved. What aggressive but sensible investment approach should I take?",
            
            # Financial Abundance Scenarios
            "I received a bonus of ${amount}. Should I add it to my existing investments or try something new?",
            "My portfolio has grown to ${portfolio_size}. At what point should I consider hiring a financial advisor?",
            "I've maxed out my 401(k) and IRA and still have ${amount} to invest monthly. Where should I put it?",
            "After selling my business for ${amount}, how should I invest this windfall for long-term growth and tax efficiency?",
            
            # Life Transition Scenarios
            "I'm {age} and just inherited ${amount}. How should I invest it differently than my existing ${portfolio_size} portfolio?",
            "I'm about to retire with ${portfolio_size}. How should I restructure my investments to generate ${amount} monthly income?",
            "I'm a single parent with ${amount} to invest and need to prioritize both safety and my child's college fund. What's my best approach?",
            "After divorce, I'm starting over with ${amount}. What investment strategy balances my need for growth and security?"
        ]

        # Market Analysis Templates
        self.market_analysis_templates = [
            # Basic Analysis
            "What does a P/E ratio of {pe_ratio} tell us about {company}?",
            "How do I interpret {company}'s dividend yield of {dividend_yield}%?",
            "What's the significance of {company}'s market cap of ${market_cap}?",
            
            # Intermediate Analysis
            "Compare {company1} and {company2}'s financial performance using their key metrics: {metrics}",
            "How would {economic_event} impact {sector} stocks, particularly {company}?",
            "Analyze {company}'s competitive position given {market_share}% market share and {growth_rate}% growth",
            
            # Advanced Analysis
            "Given {company}'s latest earnings report showing {metrics}, evaluate its growth potential versus {competitor}",
            "How would you value {company} using DCF analysis with {growth_rate}% growth and {wacc}% WACC?",
            "Has {company}'s recently acquired {target_company} for ${deal_value}M considering {synergies}",
            "Has {company} been undervalued or overvalued based on {ratios} compared to industry peers?",
            "How would you assess {company}'s industry position and growth prospects in {industry}?"
        ]

        # Corporate Finance Templates
        self.corporate_finance_templates = [
            # Basic Corporate Finance
            "How do you calculate a company's WACC?",
            "What's the significance of a {de_ratio} debt-to-equity ratio?",
            "How does a company's credit rating affect its cost of capital?",
            
            # Intermediate Corporate Finance
            "Calculate {company}'s enterprise value given: Market Cap=${market_cap}M, Debt=${debt}M, Cash=${cash}M",
            "How should {company} finance a ${project_cost}M project given {metrics}?",
            "Evaluate {company}'s dividend policy with {payout_ratio}% payout ratio and ${fcf}M free cash flow",
            
            # Advanced Corporate Finance
            "Calculate the enterprise value of {company} using DCF analysis, given: "
            "Free Cash Flow=${fcf}M, Growth Rate={growth}%, WACC={wacc}%",
            "Analyze {company}'s optimal capital structure given ${ebit}M EBIT, {tax_rate}% tax rate, and {metrics}",
            "Value {company}'s IPO given {financials} and comparable companies trading at {multiples}"
        ]

        # Tax Strategy Templates
        self.tax_strategy_templates = [
            # Basic Tax Planning
            "How can I reduce my tax bill with a ${income} salary?",
            "What tax deductions are available for a first-time homebuyer?",
            "Should I choose a Traditional or Roth IRA given my ${income} income?",
            
            # Intermediate Tax Planning
            "With ${income} income and ${investment_income} investment gains, how can I optimize my tax situation?",
            "What's the most tax-efficient way to withdraw from my ${retirement_savings} retirement accounts?",
            "How can I use tax-loss harvesting with my ${portfolio_size} portfolio?",
            
            # Advanced Tax Planning
            "With an income of ${income} and investments generating ${investment_income} "
            "in {income_type}, what tax-optimization strategies would you recommend?",
            "How can I minimize estate taxes on ${estate_value} assets using {strategies}?",
            "What's the optimal structure for my ${business_value} business to minimize taxes while {goals}?"
        ]

    def generate_financial_profile(self) -> FinancialProfile:
        """Generate a realistic financial profile for scenarios."""
        income = random.randint(30000, 500000)
        age = random.randint(22, 65)
        savings = int(income * random.uniform(0.1, 2.0))
        
        # Generate realistic debt based on income
        debt = {}
        for debt_type, (min_val, max_val) in self.debt_types.items():
            if random.random() < 0.3:  # 30% chance of having each type of debt
                # Fix: Check if min_val is less than income before generating debt
                # And ensure min_val is always less than max_val for valid range
                if min_val < income:
                    # Set possible debt range based on income and min/max values
                    debt_min = min_val
                    debt_max = min(max_val, income)
                    
                    # Only add debt if there's a valid range
                    if debt_min < debt_max:
                        debt[debt_type] = random.randint(debt_min, debt_max)
                # For very low incomes, scale down the minimum debt value proportionally
                elif debt_type == "mortgage" and income < 50000:
                    # For low income, allow smaller mortgage amounts
                    debt[debt_type] = random.randint(30000, income)
                elif min_val > income:
                    # Skip this debt type for low income individuals
                    continue
        
        return FinancialProfile(
            income=income,
            age=age,
            savings=savings,
            debt=debt,
            dependents=random.randint(0, 3),
            risk_tolerance=random.choice(self.risk_profiles),
            investment_horizon=random.randint(5, 30),
            tax_bracket=random.choice([12, 22, 24, 32, 35, 37]),
            goals=random.sample(self.investment_goals, k=random.randint(1, 3))
        )

    def generate_questions(self) -> List[str]:
        """Generate a diverse set of finance-related questions."""
        questions = []
        
        # Distribution of question types
        distributions = {
            'personalized': 0.25,
            'investment': 0.20,
            'market': 0.20,
            'corporate': 0.15,
            'tax': 0.20
        }
        
        for _ in range(self.num_questions):
            category = random.choices(
                list(distributions.keys()),
                weights=list(distributions.values()),
                k=1
            )[0]
            
            profile = self.generate_financial_profile()
            
            if category == 'personalized':
                question = self._generate_personalized_question(profile)
            elif category == 'investment':
                question = self._generate_investment_question(profile)
            elif category == 'market':
                question = self._generate_market_question()
            elif category == 'corporate':
                question = self._generate_corporate_question()
            else:  # tax
                question = self._generate_tax_question(profile)
            
            questions.append(question)
        
        return questions

    def _generate_personalized_question(self, profile: FinancialProfile) -> str:
        """Generate a personalized financial advice question."""
        template = random.choice(self.personalized_advice_templates)
        
        # Calculate additional variables needed by some templates
        amount = int(profile.savings * random.uniform(0.2, 0.8))
        old_income = int(profile.income * random.uniform(0.7, 0.9))
        new_income = profile.income
        portfolio_size = int(profile.savings * random.uniform(1.5, 4.0))
        asset_types = ", ".join(random.sample(self.asset_classes, k=random.randint(2, 4)))
        debt_types = ", ".join(list(profile.debt.keys()) if profile.debt else ["no debt"])
        goals = ", ".join(profile.goals)
        dependents = profile.dependents
        
        # Fill template with realistic data
        return template.format(
            income=f"{profile.income:,}",
            age=profile.age,
            savings=f"{profile.savings:,}",
            debt_amount=f"{sum(profile.debt.values()):,}" if profile.debt else "0",
            debt_type=random.choice(list(profile.debt.keys())) if profile.debt else "no",
            debt_types=debt_types,
            risk_tolerance=profile.risk_tolerance,
            goal=random.choice(profile.goals),
            goals=goals,
            timeframe=profile.investment_horizon,
            amount=f"{amount:,}",
            old_income=f"{old_income:,}",
            new_income=f"{new_income:,}",
            portfolio_size=f"{portfolio_size:,}",
            asset_types=asset_types,
            dependents=dependents
        )

    def _generate_investment_question(self, profile: FinancialProfile) -> str:
        """Generate an investment strategy question."""
        template = random.choice(self.investment_strategy_templates)
        
        # Generate realistic investment scenarios
        portfolio_size = profile.savings * random.uniform(1.5, 4.0)
        investment_amount = min(portfolio_size * random.uniform(0.1, 0.3), profile.savings)
        
        factors = random.sample(["Value", "Growth", "Momentum", "Quality", "Size", "Low Volatility"], k=random.randint(2, 4))
        assets = random.sample(self.asset_classes, k=random.randint(3, 6))
        debt_amount = sum(profile.debt.values()) if profile.debt else 0
        
        # Fill in template with all possible variables
        return template.format(
            amount=f"{int(investment_amount):,}",
            portfolio_size=f"{int(portfolio_size):,}",
            assets=", ".join(assets),
            market_condition=random.choice(self.market_conditions),
            timeframe=profile.investment_horizon,
            strategy=random.choice(["Value", "Growth", "Income", "Blend"]),
            economic_indicators=", ".join(random.sample([
                "rising inflation", "declining interest rates", "strong GDP growth",
                "weak consumer spending", "high unemployment", "strong dollar"
            ], k=2)),
            factors=", ".join(factors),
            num_assets=len(assets),
            debt_amount=f"{debt_amount:,}",
            age=profile.age,
            savings=f"{profile.savings:,}"  # Added missing savings parameter
        )

    def _generate_market_question(self) -> str:
        """Generate a market analysis question."""
        template = random.choice(self.market_analysis_templates)
        
        # Generate realistic market metrics
        company = random.choice([
            "Apple", "Microsoft", "Amazon", "Google", "Meta",
            "Tesla", "Nvidia", "JPMorgan", "Johnson & Johnson",
            # ...existing company list...
        ])
        
        metrics = {
            "revenue_growth": f"{random.uniform(5, 30):.1f}%",
            "profit_margin": f"{random.uniform(10, 40):.1f}%",
            "pe_ratio": f"{random.uniform(15, 50):.1f}",
            "market_cap": f"{random.randint(50, 2000)}B",
            "dividend_yield": f"{random.uniform(1, 5):.2f}%"
        }
        
        competitor_list = [
            "Apple", "Microsoft", "Amazon", "Google", "Meta",
            # ...existing competitor_list...
        ]
        
        # Generate company1 and company2 for comparison questions
        company1 = company
        company2 = random.choice([c for c in competitor_list if c != company])
        
        # Generate additional financial parameters
        growth_rate = f"{random.uniform(5, 25):.1f}"
        market_share = f"{random.uniform(10, 50):.1f}"
        wacc = f"{random.uniform(8, 15):.1f}"
        
        # Generate additional context parameters
        economic_event = random.choice([
            "Federal Reserve rate hike",
            "new trade agreement",
            "global supply chain disruption",
            "major industry regulation change",
            "economic recession concerns",
            "inflation surge"
        ])
        
        sector = random.choice([
            "Technology", "Financial Services", "Healthcare", 
            "Consumer Goods", "Energy", "Industrial", "Utilities",
            "Real Estate", "Telecommunications", "Materials"
        ])
        
        target_company = random.choice([c for c in competitor_list if c != company])
        deal_value = random.randint(500, 10000)
        synergies = random.choice([
            "cost reduction", "market expansion", "technology acquisition",
            "vertical integration", "geographical expansion"
        ])
        
        industry = sector  # Alias for sector
        
        # Format the template with all possible parameters
        return template.format(
            company=company,
            company1=company1,
            company2=company2,
            competitor=random.choice([c for c in competitor_list if c != company]),
            metrics=", ".join([f"{k.replace('_', ' ')}: {v}" for k, v in metrics.items()]),
            pe_ratio=metrics["pe_ratio"],
            dividend_yield=metrics["dividend_yield"],
            market_cap=metrics["market_cap"],
            growth_rate=growth_rate,
            market_share=market_share,
            wacc=wacc,
            economic_event=economic_event,
            sector=sector,
            target_company=target_company,
            deal_value=deal_value,
            synergies=synergies,
            ratios=", ".join([
                f"P/E: {metrics['pe_ratio']}", 
                f"P/S: {random.uniform(1, 10):.1f}", 
                f"D/E: {random.uniform(0.1, 2):.2f}"
            ]),
            industry=industry
        )

    def _generate_corporate_question(self) -> str:
        """Generate a corporate finance question."""
        template = random.choice(self.corporate_finance_templates)
        
        # Generate realistic corporate finance metrics
        financials = {
            "market_cap": random.randint(1000, 50000),
            "debt": random.randint(100, 5000),
            "cash": random.randint(50, 2000),
            "fcf": random.randint(100, 2000),
            "ebit": random.randint(200, 3000),
        }
        
        # Additional parameters needed for some templates
        company1 = random.choice([
            "Apple", "Microsoft", "Amazon", "Google", "Meta",
            "Tesla", "Nvidia", "JPMorgan", "Johnson & Johnson"
        ])
        company2 = random.choice([
            "Intel", "AMD", "IBM", "Oracle", "Salesforce",
            "Goldman Sachs", "Citigroup", "Pfizer", "Merck"
        ])
        project_cost = random.randint(50, 500)
        interest_coverage = round(random.uniform(1.5, 8.0), 1)
        synergies = random.randint(50, 200)
        multiples = f"P/E: {random.uniform(15, 30):.1f}x, EV/EBITDA: {random.uniform(8, 16):.1f}x"
        
        return template.format(
            company=company1,
            company1=company1,
            company2=company2,
            market_cap=financials["market_cap"],
            debt=financials["debt"],
            cash=financials["cash"],
            fcf=financials["fcf"],
            ebit=financials["ebit"],
            growth=f"{random.uniform(5, 20):.1f}",
            wacc=f"{random.uniform(8, 15):.1f}",  # Fixed double colon issue here
            tax_rate=f"{random.uniform(20, 35):.1f}",
            de_ratio=f"{random.uniform(0.5, 2.5):.2f}",
            payout_ratio=f"{random.uniform(20, 80):.1f}",
            project_cost=project_cost,
            interest_coverage=interest_coverage,
            synergies=synergies,
            multiples=multiples,
            financials=f"Revenue: ${random.randint(500, 5000)}M, EBITDA: ${random.randint(100, 1000)}M",
            metrics=", ".join([
                f"ROE: {random.uniform(10, 30):.1f}%",
                f"Operating Margin: {random.uniform(15, 40):.1f}%",
                f"Current Ratio: {random.uniform(1.2, 3.0):.2f}"
            ])
        )

    def _generate_tax_question(self, profile: FinancialProfile) -> str:
        """Generate a tax strategy question."""
        template = random.choice(self.tax_strategy_templates)
        
        # Generate realistic tax scenarios
        investment_income = profile.savings * random.uniform(0.05, 0.15)
        retirement_savings = profile.savings * random.uniform(2, 5)
        business_value = profile.income * random.uniform(2, 10)
        
        # Additional parameters for some templates
        account_types = ", ".join(random.sample([
            "401(k)", "Traditional IRA", "Roth IRA", "HSA", 
            "529 Plan", "Brokerage account"
        ], k=random.randint(2, 4)))
        
        strategy = random.choice([
            "tax-loss harvesting", "charitable donations", 
            "retirement account contributions", "qualified business income deduction"
        ])
        
        gains = int(investment_income * random.uniform(0.5, 2.0))
        
        return template.format(
            income=f"{profile.income:,}",
            investment_income=f"{int(investment_income):,}",
            retirement_savings=f"{int(retirement_savings):,}",
            income_type=random.choice([
                "dividend income", "capital gains", "rental income",
                "interest income", "mixed investment income"
            ]),
            tax_bracket=profile.tax_bracket,
            estate_value=f"{int(profile.savings * random.uniform(5, 10)):,}",
            strategies=", ".join(random.sample([
                "trusts", "gifting", "charitable donations",
                "family limited partnerships", "life insurance"
            ], k=random.randint(2, 3))),
            portfolio_size=f"{int(profile.savings * random.uniform(1.5, 3)):,}",
            business_value=f"{int(business_value):,}",
            account_types=account_types,
            strategy=strategy,
            gains=f"{gains:,}",
            goals=random.choice(["maximizing growth", "minimizing taxes", "ensuring business succession"])
        )

    def write_questions_to_file(self, questions: List[str]) -> None:
        """Write generated questions to file with formatting."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for i, question in enumerate(questions, 1):
                f.write(f"Question {i}:\n{question}\n\nAnswer:\n\n")
                f.write("-" * 80 + "\n\n")

    def run(self) -> None:
        """Execute the full question generation process."""
        logger.info(f"Generating {self.num_questions} advanced finance questions")
        
        with Progress() as progress:
            task = progress.add_task("[green]Generating questions...", total=self.num_questions)
            
            questions = self.generate_questions()
            progress.update(task, advance=self.num_questions)
        
        self.write_questions_to_file(questions)
        logger.info(f"Successfully generated questions to: {self.output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate advanced finance questions for training data"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/home/zahemen/datasets/advanced_finance_questions.txt",
        help="Output file path"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of questions to generate"
    )
    
    args = parser.parse_args()
    
    generator = AdvancedFinanceQuestionGenerator(args.output, args.count)
    generator.run()

if __name__ == "__main__":
    main()
