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
           (10000, 30000) , (30001, 50000), (50001, 75000), (75001, 100000),
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
            "I make ${income} per month after taxes. How should I actually divide this between bills, saving some money, and maybe investing a little?",
            "I've managed to save ${amount} and it's just sitting in my checking account. What should a regular person like me do with it?",
            "I earn ${income} a year and have ${debt_amount} in {debt_type} that's stressing me out. How can I build an emergency fund while dealing with this debt?",
            "I'm {age} and only have ${savings} put away so far. Am I behind on retirement savings and what can I do about it?",
            
            # Intermediate Planning
            "I take home ${income} annually with ${savings} in the bank. How should I budget each month to achieve my {goal} within {timeframe} years without feeling deprived?",
            "I'm {age}, make ${income}/year, and still have ${debt_amount} in {debt_type} hanging over me. What's a practical way to get debt-free while still saving something for retirement?",
            "My salary just went up from ${old_income} to ${new_income} after taxes. How should I adjust my spending and saving so I don't waste this opportunity?",
            
            # Complex Planning
            "I bring home ${income}, have ${savings} saved up, and owe ${debt_amount} across {debt_types}. How should a normal person like me handle all this while working toward {goals}?",
            "I'm {age} with {dependents} kids, making ${income}, and consider myself {risk_tolerance} with money. How do I juggle saving for college, retirement, and paying down my ${debt_amount} mortgage?",
            "I have a ${portfolio_size} investment portfolio and a {risk_tolerance} approach to risk. Should I change how much I have in {asset_types} for my {timeframe}-year timeline to reach {goal}?",
            
            # Career-based Financial Planning
            "I'm {age} and thinking about quitting my ${income} job to start my own business. With ${savings} saved up and {dependents} people depending on me, what should I consider financially?",
            "I'm switching from a steady paycheck to contract work making around ${income}. How should a regular person handle taxes, health insurance and retirement with ${savings} in savings and a {risk_tolerance} personality?",
            
            # Life Event Financial Planning
            "I want to buy a house in {timeframe} years and have ${savings} saved. While making ${income} and still paying off ${debt_amount} in student loans, what's a realistic plan?",
            "Baby on the way in 6 months! We make ${income} combined and have ${savings} saved up. How should we rethink our finances with {goal} in mind and {timeframe}-year timeline?",
            "I'm {age} and just inherited ${savings} from a relative. As someone making ${income} with a {risk_tolerance} approach to money, what should I actually do with this inheritance?",
            
            # Geographic Financial Planning
            "I'm moving to an expensive city for a job paying ${income}. With ${savings} saved and still owing ${debt_amount} on my old mortgage, how do I adjust my budget?",
            "I'm hoping to retire in another country in {timeframe} years. How should I adjust my ${portfolio_size} investments to handle currency risks with my {risk_tolerance} risk tolerance?",
            
            # Crypto-Focused Planning
            "I'm curious about Bitcoin and have ${amount} I could invest. Is crypto something a regular person like me should even consider with my ${income} salary?",
            "I've heard people make money with crypto. With ${savings} saved and a {risk_tolerance} approach, should I put some money in Bitcoin or Ethereum alongside my regular investments?",
            "Between my ${portfolio_size} traditional investments and growing interest in crypto, how much should someone making ${income} realistically put into digital currencies?",
            "I have ${amount} I want to invest in crypto, but I'm {age} with {timeframe} years until retirement. Is this too risky for my situation?",
            
            # Stock Market Planning
            "I want to try buying individual stocks with ${savings} of my savings. As someone making ${income}, should I do this or stick with safer options?",
            "I keep hearing about index funds versus picking stocks. With my ${income} income and {risk_tolerance} personality, which approach makes more sense?",
            "My friend says I should put my ${amount} bonus into tech stocks. With my {goal} and {timeframe}-year timeline, is this actually a good idea?",
            "I'm {age} with ${savings} saved and wondering if dividend stocks are better than growth stocks for someone with my {risk_tolerance} approach to investing?"
        ]

        # Investment Strategy Templates
        self.investment_strategy_templates = [
            # Basic Investment
            "I'm a complete beginner and have ${amount} saved up. What's the easiest way to start investing?",
            "Everyone talks about stocks vs. index funds. With my ${amount} savings, which one makes more sense for a regular person like me?",
            "I keep hearing about dividend stocks. How can I build a simple dividend portfolio with my ${amount} savings?",
            "After paying rent and bills, I only have ${amount} left each month. Is it even worth investing such a small amount or should I just save it?",
            "I've never bought stocks before and have ${amount} I want to invest. What's the least risky way for a first-timer?",
            
            # Intermediate Investment
            "My portfolio has {assets} but I'm confused about rebalancing during this {market_condition}. What should a regular investor like me do?",
            "I've saved up ${portfolio_size} for retirement. How should I spread this money around for a {timeframe}-year timeline?",
            "I've got ${amount} and want to try diversifying across {num_assets} different things like stocks, bonds, and maybe some crypto. How would you split it up?",
            "I contribute ${amount} monthly to my 401(k) but it doesn't seem to be growing. What am I doing wrong as an average investor?",
            "I'm torn between paying off my ${debt_amount} credit card debt and investing ${amount} in stocks. Which should come first?",
            
            # Advanced Investment
            "My ${portfolio_size} portfolio follows a {strategy} approach, but with this {market_condition} and {economic_indicators}, should I change anything?",
            "I've been reading about factor investing. How can someone like me implement this strategy with ${amount} focusing on {factors}?",
            "Are options trading strategies realistic for a regular person with a ${portfolio_size} portfolio during this {market_condition}, or is that too risky?",
            
            # Crypto-Focused Questions
            "Is it smart for someone like me to put some of my ${amount} savings into Bitcoin or Ethereum, or is crypto still too risky for regular folks?",
            "I've heard about staking crypto to earn passive income. Would it make sense to move ${amount} from my savings into staking platforms?",
            "Between established cryptocurrencies and newer altcoins, where should a cautious investor with ${amount} focus?",
            "How much of my ${portfolio_size} portfolio should realistically go into crypto as part of a balanced strategy?",
            
            # Stock-Focused Questions
            "I want to try picking individual stocks with ${amount}. How many different companies should a beginner investor buy?",
            "Everyone's talking about tech stocks. Should I put my ${amount} savings in big tech companies or spread it around different sectors?",
            "Between growth stocks and value stocks, what makes more sense for my ${amount} investment and {timeframe}-year horizon?",
            "How do I research stocks properly before investing my hard-earned ${amount}? What numbers should regular people focus on?",
            
            # Financial Hardship Scenarios
            "I got laid off and only have ${amount} in savings. Should I keep it all as emergency cash or still invest some portion?",
            "After paying medical bills, I can only save ${amount} monthly. Is there any investment strategy that makes sense for my situation?",
            "Between two part-time jobs, I manage to save ${amount} each month. What's the best way to invest when I'm stretched thin?",
            "I'm {age} years old with only ${savings} saved for retirement. What catching-up strategy would work for a regular person in my shoes?",
            
            # Financial Abundance Scenarios
            "I just got a ${amount} work bonus. Should I add it to my existing investments or try something new like crypto?",
            "My portfolio surprisingly grew to ${portfolio_size}. At what point should an average person like me consider getting professional advice?",
            "I've maxed out my retirement accounts and still have ${amount} left monthly. Where should a regular person put their extra money?",
            "I sold my house and have ${amount} extra after buying a new one. How should I invest this windfall as someone who isn't a finance expert?",
            
            # Life Transition Scenarios
            "I'm {age} and just inherited ${amount} from my grandparent. How should I invest it differently than my current ${portfolio_size} savings?",
            "I'm preparing to retire with ${portfolio_size}. How do I restructure my investments to get ${amount} monthly income without complicated strategies?",
            "As a single parent with ${amount} to invest, how can I balance building both an emergency fund and my child's college savings?",
            "After my divorce, I'm basically starting over with ${amount}. What simple investment approach gives me both growth potential and security?"
        ]
        # Market Analysis Templates
        self.market_analysis_templates = [
            # Basic Analysis
            "I noticed {company} has a P/E ratio of {pe_ratio}. Is this considered overvalued for their sector?",
            "As an income investor, is {company}'s {dividend_yield}% dividend yield sustainable based on their financials?",
            "I'm researching {company} which has a market cap of ${market_cap}. What does this tell me about its stability and growth potential?",
            
            # Intermediate Analysis
            "I'm deciding between investing in {company1} or {company2}. How do they compare in terms of {metrics}?",
            "How might the recent {economic_event} affect my investments in {sector} companies like {company}?",
            "With {company} holding {market_share}% market share and growing at {growth_rate}%, is it likely to maintain its competitive edge?",
            
            # Advanced Analysis
            "After reading {company}'s quarterly report showing {metrics}, should I consider it a better long-term investment than {competitor}?",
            "I've tried valuing {company} using DCF with {growth_rate}% growth and {wacc}% WACC, but is this approach reliable given current market conditions?",
            "{company} just acquired {target_company} for ${deal_value}M citing {synergies}. Will this actually create shareholder value?",
            "Analysts say {company} is trading at {ratios} - are they getting a fair valuation compared to similar companies in their space?",
            "I've been tracking {company} in the {industry} sector. Given current trends, does it make sense to invest for the next 3-5 years?"
        ]

        # Corporate Finance Templates
        self.corporate_finance_templates = [
            # Basic Corporate Finance
            "As an average investor, why should I care about {company}'s WACC and how would I calculate it?",
            "When reading financial news about {company}, what does their {de_ratio} debt-to-equity ratio tell me about their financial health?",
            "How does {company}'s recent credit rating change affect their ability to grow and potentially impact my investment?",
            
            # Intermediate Corporate Finance
            "I'm researching {company} for my portfolio - what's their true value considering Market Cap=${market_cap}M, Debt=${debt}M, and Cash=${cash}M?",
            "If {company} announced a new ${project_cost}M expansion project with {metrics}, how might they pay for it and would this benefit shareholders?",
            "As a dividend investor, should I be concerned about {company}'s {payout_ratio}% payout ratio with ${fcf}M free cash flow?",
            
            # Advanced Corporate Finance
            "I'm considering investing in {company} long-term. Using their Free Cash Flow=${fcf}M, Growth Rate={growth}%, and WACC={wacc}%, what's a fair price for their stock?",
            "Will {company}'s financing decisions benefit or hurt shareholders given their ${ebit}M annual earnings, {tax_rate}% tax rate, and {metrics}?",
            "Is {company}'s upcoming IPO worth investing in based on {financials} when similar companies are valued at {multiples}?"
        ]

        # Tax Strategy Templates
        self.tax_strategy_templates = [
            # Basic Tax Planning
            "What tax strategies should someone with ${income} annual income consider to reduce their tax burden?",
            "How can I optimize my tax situation with an annual income of ${income}?",
            "What tax-advantaged accounts are recommended for someone making ${income} per year?",
            
            # Intermediate Tax Planning
            "If I make ${income} from my job and ${investment_income} from {income_type}, what tax planning strategies should I consider?",
            "What's the most tax-efficient way to withdraw from my ${retirement_savings} in retirement?",
            "How should I manage taxes on my ${portfolio_size} investment portfolio consisting of different asset types?",
            
            # Advanced Tax Planning
            "What tax strategies work best for someone with ${income} salary and ${investment_income} in {income_type}?",
            "How can I protect my family's financial future and minimize taxes on my ${estate_value} estate using {strategies}?",
            "I own a small business worth ${business_value} - what tax planning strategies should I consider for both business and personal taxes?"
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
            # Tech
            "Apple", "Microsoft", "Amazon", "Google", "Meta", "Tesla", "Nvidia",
            # Financial
            "JPMorgan", "Goldman Sachs", "Visa", "Bank of America",
            # Healthcare
            "Johnson & Johnson", "UnitedHealth", "Pfizer",
            # Retail
            "Walmart", "Target", "Home Depot", "Costco",
            # Energy
            "ExxonMobil", "Chevron", "Shell", "BP",
            # Industrial
            "Boeing", "Caterpillar", "3M", "General Electric",
            # Consumer Goods
            "Procter & Gamble", "Coca-Cola", "PepsiCo", "Nike",
            # International
            "Toyota", "Samsung", "Alibaba", "HSBC", "Volkswagen", "Tencent", "Nestlé"
        ])

        company2 = random.choice([
            # Tech
            "Intel", "AMD", "IBM", "Oracle", "Salesforce", "Adobe", "Dell", "PayPal",
            # Financial
            "Citigroup", "Wells Fargo", "American Express", "Morgan Stanley", "Mastercard",
            # Healthcare
            "Merck", "Novartis", "Eli Lilly", "AstraZeneca", "CVS Health",
            # Retail
            "Lowe's", "TJX", "Dollar General", "Best Buy", "Kroger",
            # Energy
            "ConocoPhillips", "Occidental", "Duke Energy", "NextEra Energy",
            # Industrial
            "Honeywell", "Lockheed Martin", "Union Pacific", "Deere & Company",
            # Consumer Goods
            "Unilever", "Colgate-Palmolive", "Kraft Heinz", "Adidas",
            # Entertainment/Media
            "Disney", "Netflix", "Warner Bros", "Spotify", "Sony",
            # International
            "Honda", "Siemens", "BASF", "JD.com", "Reliance Industries", "Roche"
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
        default="/home/zahemen/datasets/advanced_finance_questions_.txt",
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
