def format_verdict(final_verdict, reasoning, confidence, investor_outlook):
    return (
        f"📈 Final Verdict: {final_verdict}\n"
        f"💬 Reasoning: {reasoning}\n"
        f"📊 Confidence Level: {confidence}\n"
        f"💰 Investor Outlook: {investor_outlook}\n"
    )