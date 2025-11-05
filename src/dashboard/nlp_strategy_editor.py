"""
Streamlit Dashboard for NLP Strategy Editor
Create trading strategies using natural language
"""

import streamlit as st
from src.nlp_strategy.strategy_pipeline import NLPStrategyPipeline
import os
import sys

st.set_page_config(page_title="NLP Strategy Editor", layout="wide", page_icon="🤖")

st.title("🤖 NLP Strategy Editor")
st.markdown("**Create trading strategies in natural language powered by Claude AI**")

# Sidebar
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Anthropic API Key", 
    type="password", 
    value=os.environ.get('ANTHROPIC_API_KEY', ''),
    help="Enter your Anthropic API key to use Claude AI"
)

if api_key:
    os.environ['ANTHROPIC_API_KEY'] = api_key

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This tool uses Claude AI to:
    - Parse natural language strategy descriptions
    - Generate Backtrader Python code
    - Validate strategies
    - Enable backtesting
    """
)

# Main content
st.header("1️⃣ Describe Your Strategy")

# Example strategies
example_strategies = {
    "Custom": "",
    "RSI Oversold/Overbought": """Je veux une stratégie qui achète quand le RSI tombe en dessous de 30 
    et vend quand il dépasse 70. Utilise un stop loss de 2% et take profit de 4%. 
    Trade sur timeframe 1h.""",
    
    "EMA Crossover with Volume": """Créer une stratégie qui achète quand l'EMA 10 croise au-dessus 
    de l'EMA 50 ET que le volume est supérieur à la moyenne. Sortir quand l'EMA 10 repasse 
    sous l'EMA 50. Stop loss 3%, take profit 6%. Timeframe 4h.""",
    
    "Bollinger Bands Bounce": """Je veux acheter quand le prix touche la bande de Bollinger inférieure 
    et que le RSI < 40. Vendre quand le prix atteint la bande médiane. Stop loss 1.5%, 
    take profit 3%. Timeframe 1h.""",
    
    "MACD Momentum": """Stratégie qui entre en position longue quand la ligne MACD croise au-dessus 
    du signal ET que le prix est au-dessus de l'EMA 200. Sortir quand MACD recroise en dessous 
    du signal. Stop loss 2.5%, take profit 5%. Timeframe 1h."""
}

selected_example = st.selectbox(
    "📚 Choose an example or write your own",
    list(example_strategies.keys())
)

default_text = example_strategies[selected_example]

strategy_description = st.text_area(
    "Strategy Description",
    value=default_text,
    height=150,
    placeholder="Ex: Je veux une stratégie qui achète quand le RSI est en dessous de 30...",
    help="Describe your trading strategy in natural language (English or French)"
)

col1, col2 = st.columns([1, 3])

with col1:
    generate_button = st.button("🚀 Generate Strategy", type="primary", use_container_width=True)

# Generate strategy
if generate_button:
    if not api_key:
        st.error("❌ Please provide an Anthropic API Key in the sidebar")
    elif not strategy_description:
        st.error("❌ Please provide a strategy description")
    else:
        with st.spinner("🤖 Generating strategy with Claude AI..."):
            try:
                pipeline = NLPStrategyPipeline(api_key)
                result = pipeline.create_strategy_from_text(strategy_description)
                
                st.session_state['result'] = result
                st.success("✅ Strategy generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Display results
if 'result' in st.session_state:
    result = st.session_state['result']
    strategy = result['strategy']
    
    st.markdown("---")
    st.header("2️⃣ Generated Strategy")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Strategy Name", strategy.name)
    with col2:
        st.metric("Timeframe", strategy.timeframe)
    with col3:
        st.metric("Indicators", len(strategy.indicators))
    with col4:
        risk_ratio = strategy.risk_management.get('take_profit_pct', 0) / max(strategy.risk_management.get('stop_loss_pct', 1), 0.001)
        st.metric("Risk/Reward", f"{risk_ratio:.2f}")
    
    # Issues
    if result['issues']:
        st.warning("⚠️ Issues Detected:")
        for issue in result['issues']:
            st.write(f"- {issue}")
    else:
        st.success("✅ No issues detected")
    
    # Strategy Details
    with st.expander("📋 Strategy Details", expanded=True):
        st.markdown(f"**Description:** {strategy.description}")
        
        st.markdown("**Entry Conditions:**")
        for cond in strategy.entry_conditions:
            st.write(f"- {cond}")
        
        st.markdown("**Exit Conditions:**")
        for cond in strategy.exit_conditions:
            st.write(f"- {cond}")
        
        st.markdown("**Indicators:**")
        st.write(", ".join(strategy.indicators))
        
        st.markdown("**Risk Management:**")
        rm = strategy.risk_management
        st.json(rm)
    
    # Code Display
    st.markdown("---")
    st.header("3️⃣ Generated Code")
    
    tab1, tab2 = st.tabs(["📄 View Code", "📥 Download"])
    
    with tab1:
        st.code(result['code'], language='python')
    
    with tab2:
        st.download_button(
            label="📥 Download Strategy Code",
            data=result['code'],
            file_name=f"{strategy.name.lower().replace(' ', '_')}.py",
            mime="text/plain",
            use_container_width=True
        )
        
        st.info(f"💾 Strategy also saved to: `{result['filename']}`")
    
    # Backtesting Section
    st.markdown("---")
    st.header("4️⃣ Backtest Strategy")
    
    st.info("⚠️ Backtesting integration coming soon! The generated strategy can be tested using the existing backtesting framework.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "BNB/USDT"])
    with col2:
        days = st.slider("Period (days)", 30, 365, 90)
    with col3:
        initial_capital = st.number_input("Initial Capital ($)", 1000, 100000, 10000, step=1000)
    
    if st.button("📊 Run Backtest", disabled=True):
        st.info("Backtesting feature will be integrated in the next phase")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and Claude AI</p>
        <p><small>NLP Strategy Editor v1.0</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
