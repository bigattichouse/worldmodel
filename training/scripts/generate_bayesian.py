#!/usr/bin/env python3
"""
Generate Bayesian reasoning training examples.

Covers: Bayes' theorem, conditional probability, prior/posterior updating,
naive Bayes classification, Bayesian A/B testing, and belief updating chains.

Usage:
    python training/scripts/generate_bayesian.py --output training/datasets/bayesian/basic.jsonl --count 100
"""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import run_once, PythonExecutor


def execute(code: str) -> str:
    return run_once(code).output_text()


def make_example(ex_id, category, difficulty, query, think, model_text, code):
    output = execute(code)
    parts = []
    if think:
        parts.append(f"<think>\n{think.strip()}\n</think>")
    if model_text:
        parts.append(f"<model>\n{model_text.strip()}\n</model>")
    parts.append(f"<code>\n{code.strip()}\n</code>")
    parts.append(f"<output>\n{output.strip()}\n</output>")
    return {"id": ex_id, "category": category, "difficulty": difficulty,
            "query": query, "response": "\n".join(parts)}


# ─── Generators ───────────────────────────────────────────────────────────────

def gen_medical_test(rng):
    """Classic base-rate fallacy / medical test problem"""
    diseases = [
        ("rare cancer", 0.005, 0.99, 0.95),
        ("diabetes", 0.08, 0.92, 0.88),
        ("hypertension", 0.15, 0.85, 0.80),
        ("flu", 0.10, 0.90, 0.85),
    ]
    name, prevalence, sensitivity, specificity = rng.choice(diseases)
    # Jitter slightly
    sensitivity = round(sensitivity + rng.uniform(-0.03, 0.03), 2)
    specificity = round(specificity + rng.uniform(-0.03, 0.03), 2)

    query = (
        f"A test for {name} has sensitivity {sensitivity:.0%} and specificity {specificity:.0%}. "
        f"The disease affects {prevalence:.1%} of the population. "
        f"A patient tests positive. What is the probability they actually have {name}?"
    )
    think = (
        "Apply Bayes' theorem: P(disease|positive) = P(positive|disease)·P(disease) / P(positive).\n"
        "P(positive) = sensitivity·prevalence + (1-specificity)·(1-prevalence)."
    )
    code = (
        f"prevalence = {prevalence}   # P(disease)\n"
        f"sensitivity = {sensitivity}  # P(positive | disease)\n"
        f"specificity = {specificity}  # P(negative | no disease)\n\n"
        f"# P(positive) = P(pos|disease)*P(disease) + P(pos|no disease)*P(no disease)\n"
        f"p_pos = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)\n\n"
        f"# Bayes: P(disease | positive)\n"
        f"p_disease_given_pos = (sensitivity * prevalence) / p_pos\n\n"
        f"print(f'P(positive): {{p_pos:.4f}}')\n"
        f"print(f'P(disease | positive test): {{p_disease_given_pos:.4f}} = {{p_disease_given_pos:.1%}}')\n"
        f"print(f'False positive rate: {{1 - p_disease_given_pos:.1%}}')"
    )
    return query, think, "", code


def gen_bayesian_update(rng):
    """Sequential belief updating: observe evidence, update prior"""
    # Coin bias estimation
    true_bias = rng.choice([0.3, 0.4, 0.6, 0.7, 0.8])
    n_flips = rng.randint(5, 15)
    heads = sum(1 for _ in range(n_flips) if rng.random() < true_bias)
    tails = n_flips - heads

    query = (
        f"A coin is flipped {n_flips} times, getting {heads} heads and {tails} tails. "
        f"Starting with a uniform prior, use Bayesian updating to find the posterior "
        f"distribution of the coin's bias."
    )
    think = (
        "With a uniform prior and binomial likelihood, the posterior is Beta(α + heads, β + tails) "
        "where α=β=1 for the uniform prior. The posterior mean is (1+heads)/(2+total)."
    )
    model_text = "Prior: Beta(1,1). Likelihood: Binomial. Posterior: Beta(1+H, 1+T)."
    code = (
        f"from scipy.stats import beta\n"
        f"import numpy as np\n\n"
        f"heads = {heads}\n"
        f"tails = {tails}\n\n"
        f"# Prior: Beta(1,1) = Uniform\n"
        f"alpha_prior, beta_prior = 1, 1\n\n"
        f"# Posterior: Beta(alpha + heads, beta + tails)\n"
        f"alpha_post = alpha_prior + heads\n"
        f"beta_post = beta_prior + tails\n\n"
        f"posterior = beta(alpha_post, beta_post)\n"
        f"mean = posterior.mean()\n"
        f"ci_low, ci_high = posterior.ppf(0.025), posterior.ppf(0.975)\n\n"
        f"print(f'Observations: {{heads}} heads, {{tails}} tails')\n"
        f"print(f'Posterior: Beta({{alpha_post}}, {{beta_post}})')\n"
        f"print(f'Posterior mean (estimated bias): {{mean:.3f}}')\n"
        f"print(f'95% credible interval: [{{ci_low:.3f}}, {{ci_high:.3f}}]')"
    )
    return query, think, model_text, code


def gen_naive_bayes(rng):
    """Naive Bayes classifier for text-like features"""
    # Spam detection
    spam_keywords = ["money", "prize", "click", "free", "win"]
    ham_keywords = ["meeting", "project", "report", "lunch", "schedule"]

    # Generate a simple "email"
    n_spam = rng.randint(2, 4)
    n_ham = rng.randint(0, 2)
    email_words = (rng.sample(spam_keywords, n_spam) +
                   rng.sample(ham_keywords, min(n_ham, len(ham_keywords))))
    rng.shuffle(email_words)

    query = (
        f"An email contains the words: {email_words}. "
        f"Using Naive Bayes, classify it as spam or ham. "
        f"Prior: P(spam)=0.4. "
        f"Word likelihoods given spam: {{'money':0.8,'prize':0.75,'click':0.7,'free':0.85,'win':0.78,'meeting':0.05,'project':0.03,'report':0.04,'lunch':0.06,'schedule':0.04}}. "
        f"Word likelihoods given ham: {{'money':0.1,'prize':0.05,'click':0.15,'free':0.12,'win':0.08,'meeting':0.6,'project':0.7,'report':0.65,'lunch':0.55,'schedule':0.5}}."
    )
    think = (
        "Naive Bayes: P(spam|words) ∝ P(spam) × Π P(word|spam). "
        "Compare log-probabilities for numerical stability."
    )
    code = (
        f"import math\n\n"
        f"email = {email_words}\n"
        f"p_spam_prior = 0.4\n"
        f"p_ham_prior = 0.6\n\n"
        f"spam_likelihoods = {{'money':0.8,'prize':0.75,'click':0.7,'free':0.85,'win':0.78,\n"
        f"                     'meeting':0.05,'project':0.03,'report':0.04,'lunch':0.06,'schedule':0.04}}\n"
        f"ham_likelihoods  = {{'money':0.1,'prize':0.05,'click':0.15,'free':0.12,'win':0.08,\n"
        f"                     'meeting':0.6,'project':0.7,'report':0.65,'lunch':0.55,'schedule':0.5}}\n\n"
        f"log_spam = math.log(p_spam_prior)\n"
        f"log_ham  = math.log(p_ham_prior)\n\n"
        f"for word in email:\n"
        f"    if word in spam_likelihoods:\n"
        f"        log_spam += math.log(spam_likelihoods[word])\n"
        f"        log_ham  += math.log(ham_likelihoods[word])\n"
        f"        print(f'  {{word}}: P(spam)={{spam_likelihoods[word]}}, P(ham)={{ham_likelihoods[word]}}')\n\n"
        f"# Normalize\n"
        f"log_total = math.log(math.exp(log_spam) + math.exp(log_ham))\n"
        f"p_spam_post = math.exp(log_spam - log_total)\n"
        f"p_ham_post  = math.exp(log_ham  - log_total)\n\n"
        f"print(f'\\nP(spam | email): {{p_spam_post:.3f}}')\n"
        f"print(f'P(ham  | email): {{p_ham_post:.3f}}')\n"
        f"print(f'Classification: {{\"SPAM\" if p_spam_post > 0.5 else \"HAM\"}}')"
    )
    return query, think, "", code


def gen_ab_test_bayesian(rng):
    """Bayesian A/B test: which variant is better?"""
    # Version A
    a_visitors = rng.randint(200, 500)
    a_conv_rate = round(rng.uniform(0.04, 0.12), 3)
    a_conversions = int(a_visitors * a_conv_rate)

    # Version B slightly better or worse
    b_visitors = rng.randint(200, 500)
    b_conv_rate = round(a_conv_rate + rng.uniform(-0.03, 0.04), 3)
    b_conv_rate = max(0.01, min(b_conv_rate, 0.20))
    b_conversions = int(b_visitors * b_conv_rate)

    query = (
        f"A/B test results: "
        f"Variant A: {a_conversions}/{a_visitors} conversions. "
        f"Variant B: {b_conversions}/{b_visitors} conversions. "
        f"Using Bayesian analysis, what is the probability that B is better than A?"
    )
    think = (
        "Model each variant's conversion rate as Beta distributed. "
        "Sample from both posteriors and estimate P(B > A) by Monte Carlo."
    )
    code = (
        f"import numpy as np\n"
        f"from scipy.stats import beta\n\n"
        f"# Beta posteriors from uniform prior + data\n"
        f"a_alpha = 1 + {a_conversions}\n"
        f"a_beta  = 1 + {a_visitors} - {a_conversions}\n"
        f"b_alpha = 1 + {b_conversions}\n"
        f"b_beta  = 1 + {b_visitors} - {b_conversions}\n\n"
        f"# Monte Carlo: sample from both, count P(B > A)\n"
        f"rng = np.random.default_rng(42)\n"
        f"n_samples = 100_000\n"
        f"a_samples = rng.beta(a_alpha, a_beta, n_samples)\n"
        f"b_samples = rng.beta(b_alpha, b_beta, n_samples)\n\n"
        f"p_b_better = np.mean(b_samples > a_samples)\n\n"
        f"print(f'A: {{a_alpha-1}}/{{a_alpha+a_beta-2}} conversions, mean rate = {{a_alpha/(a_alpha+a_beta):.3f}}')\n"
        f"print(f'B: {{b_alpha-1}}/{{b_alpha+b_beta-2}} conversions, mean rate = {{b_alpha/(b_alpha+b_beta):.3f}}')\n"
        f"print(f'P(B better than A): {{p_b_better:.3f}} = {{p_b_better:.1%}}')\n"
        f"print(f'Recommendation: {{\"Use B\" if p_b_better > 0.95 else \"Need more data\" if p_b_better > 0.5 else \"Keep A\"}}')"
    )
    return query, think, "", code


def gen_multistep_update(rng):
    """Multi-step: update belief as new evidence arrives"""
    # Disease screening: three sequential tests
    disease_rate = rng.choice([0.01, 0.02, 0.05])
    test_sensitivity = round(rng.uniform(0.85, 0.97), 2)
    test_specificity = round(rng.uniform(0.80, 0.95), 2)
    n_tests = 3

    query = (
        f"A disease has {disease_rate:.1%} prevalence. "
        f"A test has {test_sensitivity:.0%} sensitivity and {test_specificity:.0%} specificity. "
        f"A patient takes {n_tests} independent tests, all positive. "
        f"Show the posterior probability after each test."
    )
    think = (
        "After each positive test, use the current posterior as the new prior. "
        "This sequential updating is the core of Bayesian inference."
    )

    exec_shared = PythonExecutor()
    code1 = (
        f"sensitivity = {test_sensitivity}\n"
        f"specificity = {test_specificity}\n"
        f"prior = {disease_rate}  # baseline prevalence\n\n"
        f"def bayes_update(prior, sensitivity, specificity, positive=True):\n"
        f"    if positive:\n"
        f"        p_evidence = sensitivity * prior + (1 - specificity) * (1 - prior)\n"
        f"        return (sensitivity * prior) / p_evidence\n"
        f"    else:\n"
        f"        p_evidence = (1-sensitivity) * prior + specificity * (1 - prior)\n"
        f"        return ((1-sensitivity) * prior) / p_evidence\n\n"
        f"print(f'Prior probability: {{prior:.4f}} = {{prior:.2%}}')\n"
        f"posterior = prior\n"
        f"for i in range(1, {n_tests+1}):\n"
        f"    posterior = bayes_update(posterior, sensitivity, specificity, positive=True)\n"
        f"    print(f'After test {{i}} (positive): {{posterior:.4f}} = {{posterior:.2%}}')"
    )
    out1 = exec_shared.run(code1).output_text()

    code2 = (
        f"print(f'\\nFinal posterior after {n_tests} positive tests: {{posterior:.4f}} = {{posterior:.2%}}')\n"
        f"print(f'Odds ratio increase: {{posterior/(1-posterior) / ({disease_rate}/(1-{disease_rate})):.1f}}x')"
    )
    out2 = exec_shared.run(code2).output_text()

    response = (
        f"<think>\n{think}\n</think>\n"
        f"<code>\n{code1.strip()}\n</code>\n"
        f"<output>\n{out1.strip()}\n</output>\n"
        f"<think>\nSummarize the final posterior and quantify the belief update.\n</think>\n"
        f"<code>\n{code2.strip()}\n</code>\n"
        f"<output>\n{out2.strip()}\n</output>"
    )
    return {"id": None, "category": "bayesian", "difficulty": "intermediate",
            "query": query, "response": response}


# ─── Main ─────────────────────────────────────────────────────────────────────

GENERATORS = [
    ("medical_test",    "basic",        gen_medical_test,        0.25),
    ("belief_update",   "intermediate", gen_bayesian_update,     0.20),
    ("naive_bayes",     "intermediate", gen_naive_bayes,         0.20),
    ("ab_test",         "intermediate", gen_ab_test_bayesian,    0.20),
]


def generate_examples(count: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    idx = 0

    multistep_count = max(1, count // 6)
    single_count = count - multistep_count
    total_w = sum(g[3] for g in GENERATORS)

    for _ in range(single_count):
        r = rng.random()
        cumulative = 0.0
        chosen = GENERATORS[0]
        for gen in GENERATORS:
            cumulative += gen[3] / total_w
            if r <= cumulative:
                chosen = gen
                break
        name, difficulty, fn = chosen[0], chosen[1], chosen[2]
        try:
            query, think, model_text, code = fn(rng)
            ex = make_example(f"bayes_{idx:04d}", "bayesian", difficulty,
                               query, think, model_text, code)
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping {name} ({e})")

    for _ in range(multistep_count):
        try:
            ex = gen_multistep_update(rng)
            ex["id"] = f"bayes_{idx:04d}"
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping multistep ({e})")

    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} Bayesian examples...")
    examples = generate_examples(args.count, seed=args.seed)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Written {len(examples)} examples to {output_path}")
    errors = sum(1 for ex in examples if "<code>" in ex["response"] and "<output>" not in ex["response"])
    print("All examples validated OK." if errors == 0 else f"WARNING: {errors} missing outputs")


if __name__ == "__main__":
    main()
