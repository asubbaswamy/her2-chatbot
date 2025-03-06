from chatbot import BasicChatBot, RAGChatBot
from make_vector_db import ClinicalEmbeddings
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from scipy import stats


file_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(file_dir)

def measure_response_time(chatbot, n_trials=10):
    """Measure average response generation time for chatbot.

    Args:
        chatbot (ChatBot): Chatbot to evaluate
        n_trials (int, optional): Number of queries to average over. Defaults to 10.
    
    Returns:
        Mean, st deviation of response times in seconds.
    """
    runtimes = []
    test_query = "What are the important relationships between HER-2/neu and breast cancer?"
    
    for i in range(n_trials):
        start = time()
        response = chatbot.answer(test_query)
        stop = time()
        runtimes.append(stop-start)
    
    return np.mean(runtimes), np.std(runtimes)

def answer_in_response(answer, response):
    """Evaluates whether the brief, factual answer is contained wthin the
    response generated by the chatbot. While imperfect, this is a useful
    heuristic for checking if the chatbot response recovered the desired fact.

    Args:
        answer (String): Correct answer to question
        response (String): Chatbot response to question

    Returns:
        Boolean: Whether or not all answer words appear in response
    """
    
    answer_words = answer.split()
    for word in answer_words:
        if word not in response:
            return False
    
    return True

def evaluate_factualness(bot, qa_set):
    # get questions and generate responses
    queries = qa_set['Query'].values
    responses = []
    for query in queries:
        response = bot.answer(query).lower()
        responses.append(response)
    
    answers = qa_set['Answer'].values
    answers = [answer.lower() for answer in answers]
    
    # check if answer is contained in response
    n = len(queries)
    correctness = np.zeros(n)
    for i in range(n):
        correctness[i] = 1 * answer_in_response(answers[i], responses[i])
        
    return np.mean(correctness)
    
    
def wilson_score_interval(accuracy, n, confidence=0.95):
    """Compute error bar for proportion-based metrics

    Args:
        accuracy (float): Fraction correct
        n (int): Number of samples used to compute accuracy
        confidence (float, optional): Confidence level. Defaults to 0.95.

    Returns:
        float: The two-sided (+/-) confidence interval size
    """
    # two sided
    z = stats.norm.ppf(1 - (1-confidence)/2)
    interval = z * np.sqrt((accuracy * (1-accuracy))/n)
    
    return interval
    

def main():
    bot_names = ['Basic llama3.2', 'Basic llama3.2:1b', 'RAG llama3.2', 'RAG llama3.2:1b', 'Clinical RAG llama3.2', 'Clinical RAG llama3.2:1b']
    bots = [BasicChatBot(), BasicChatBot('llama3.2:1b'), RAGChatBot(), RAGChatBot('llama3.2:1b'),  RAGChatBot(clinical=True), RAGChatBot('llama3.2:1b', clinical=True)]
    
    facts_qa = pd.read_csv(os.path.join(project_dir, 'reference_docs') + '/factual_questions.csv')

    runtime_means = []
    runtime_stds = []
    correctness_scores = []
    n_trials = 25
    for i, bot in tqdm(enumerate(bots)):
        mean, std = measure_response_time(bot, n_trials)
        runtime_means.append(mean)
        runtime_stds.append(std)
        print("\n" + bot_names[i])
        print(f"Avg Runtime {mean} seconds; St dev {std}")
        
        correctness = evaluate_factualness(bot, facts_qa)
        correctness_scores.append(correctness)
        print(f"Correctness {correctness}")
    
    
    # plot correctness and runtimes
    plt.bar(bot_names, correctness_scores)
    correctness_error_bars = [wilson_score_interval(c, len(facts_qa)) for c in correctness_scores]
    plt.errorbar(bot_names, correctness_scores, yerr=correctness_error_bars, capsize=5, ecolor='black',elinewidth=1.5, capthick=1.5, fmt='none')
    plt.xlabel('Chatbot')
    plt.xticks(rotation=45)
    plt.ylabel('Fraction of Questions Answered Correctly')
    plt.savefig('correctness_plot.png', dpi=180, bbox_inches="tight")   
    
    # 95% confidence interval
    plt.figure()
    runtime_interval = 1.96*np.array(runtime_stds)/np.sqrt(n_trials)
    plt.bar(bot_names, runtime_means)
    plt.errorbar(bot_names, runtime_means, yerr=runtime_interval, capsize=5, ecolor='black',elinewidth=1.5, capthick=1.5, fmt='none')
    plt.xlabel('Chatbot')
    plt.xticks(rotation=45)
    plt.ylabel('Mean Response Time (seconds)')
    plt.savefig('runtime_plot.png', dpi=180, bbox_inches="tight")   


if __name__ == '__main__':
    main()