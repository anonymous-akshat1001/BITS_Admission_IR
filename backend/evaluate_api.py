import os, argparse, time, logging, traceback, warnings, requests, pandas as pd, psutil, threading, queue, nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as calculate_bertscore_hf


from dotenv import load_dotenv
import os

load_dotenv()


#Config
DATASET_PATH = os.getenv("DATASET_PATH")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 120))


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#NLTK Setup
try:
    nltk.data.find('tokenizers/punkt')
except:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    try: nltk.download('punkt', quiet=True)
    except Exception as e: logger.error(f"Failed 'punkt' download: {e}")

# Memory Tracking
def monitor_memory(proc, interval, stop_ev, q):
    peak_mem = 0
    while not stop_ev.is_set():
        try: peak_mem = max(peak_mem, proc.memory_info().rss)
        except: pass 
        time.sleep(max(interval, 0.001))
    q.put(peak_mem)

def track_peak_mem(func, *args, **kwargs):
    proc = psutil.Process(os.getpid())
    stop_ev = threading.Event()
    q = queue.Queue()
    monitor = threading.Thread(target=monitor_memory, args=(proc, 0.01, stop_ev, q), daemon=True)
    start_mem = proc.memory_info().rss
    monitor.start()
    res, err = None, None
    try: res = func(*args, **kwargs)
    except Exception as e: err = e
    finally:
        stop_ev.set()
        monitor.join(timeout=1.0)
    peak_bytes = q.get() if not q.empty() else start_mem
    peak_mb = max(start_mem, peak_bytes) / (1024*1024) if peak_bytes else 0
    if err: raise err
    return res, peak_mb

def calc_bleu(ref, cand):
    if not all(isinstance(s, str) and s for s in [ref, cand]): return 0.0
    try:
        ref_toks = [nltk.word_tokenize(ref.lower())]
        can_toks = nltk.word_tokenize(cand.lower())
        return sentence_bleu(ref_toks, can_toks, smoothing_function=SmoothingFunction().method4)
    except Exception as e: logger.warning(f"BLEU failed: {e}"); return 0.0

def calc_rouge_l(ref, cand):
    if not all(isinstance(s, str) and s for s in [ref, cand]): return 0.0
    try: return rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(ref, cand)['rougeL'].fmeasure
    except Exception as e: logger.warning(f"ROUGE-L failed: {e}"); return 0.0

def calc_bert_f1(ref, cand):
    if not all(isinstance(s, str) and s for s in [ref, cand]): return 0.0
    try: _, _, f1 = calculate_bertscore_hf([cand], [ref], lang="en", verbose=False, model_type='bert-base-uncased'); return f1.item()
    except Exception as e: logger.warning(f"BERTScore failed: {e}"); return 0.0

def run_eval(mode):
    endpoint_map = {'hybrid': "/query/hybrid/", 'hybrid-rerank': "/query/hybrid-rerank/"}
    if mode not in endpoint_map: logger.error(f"Invalid mode: {mode}"); return
    endpoint = f"{API_BASE_URL}{endpoint_map[mode]}"
    outfile = f"evaluation_results_{mode}.csv"
    logger.info(f"Starting eval: mode='{mode}', endpoint='{endpoint}', dataset='{DATASET_PATH}', output='{outfile}'")

    try:
        df = pd.read_csv(DATASET_PATH, encoding='utf-8')
        if not {'Question', 'Answer'}.issubset(df.columns):
            logger.error("'Question' and 'Answer' columns required in dataset."); return
        df.rename(columns={'Answer': 'Reference_Answer'}, inplace=True)
        df['Reference_Answer'] = df['Reference_Answer'].fillna('')
        logger.info(f"Loaded {len(df)} cases.")
    except Exception as e: logger.error(f"Dataset load error: {e}"); return

    results = []
    total = len(df)
    logger.info("Initializing BERTScore model...")
    try: calc_bert_f1("init", "init")
    except Exception as e: logger.error(f"BERTScore init failed: {e}")

    logger.info("Processing queries...")
    for idx, row in df.iterrows():
        query, ref_ans = row['Question'], row['Reference_Answer']
        logger.info(f"Query {idx + 1}/{total}: '{query[:50]}...'" )
        gen_ans, lat_ms, peak_mb, status = None, None, None, "Error"
        bleu, rouge, bert = 0.0, 0.0, 0.0

        try:
            start_time = time.perf_counter()
            resp, peak_mb = track_peak_mem(requests.post, endpoint, json={"query": query}, timeout=REQUEST_TIMEOUT)
            lat_ms = (time.perf_counter() - start_time) * 1000
            resp.raise_for_status()
            api_res = resp.json()
            gen_ans = api_res.get('answer', '')
            status = "Success"
            if gen_ans and ref_ans:
                bleu = calc_bleu(ref_ans, gen_ans)
                rouge = calc_rouge_l(ref_ans, gen_ans)
                bert = calc_bert_f1(ref_ans, gen_ans)
        except requests.exceptions.RequestException as e: logger.error(f"API Fail q={idx+1}: {e}")
        except Exception as e: logger.error(f"Error q={idx+1}: {e}\n{traceback.format_exc()}")
        finally:
             if lat_ms is None and 'start_time' in locals(): lat_ms = (time.perf_counter() - start_time) * 1000

        results.append({
            "Question": query,
            "Reference_Answer": ref_ans,
            "Generated_Answer": gen_ans,
            "Status": status,
            "Latency_ms": lat_ms,
            "Peak_Script_Memory_MB": peak_mb, 
            "BLEU_Score": bleu,
            "ROUGE-L_F1": rouge,
            "BERTScore_F1": bert
        })

    results_df = pd.DataFrame(results)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outpath = os.path.join(script_dir, outfile)
        results_df.to_csv(outpath, index=False, encoding='utf-8')
        logger.info(f"Results saved to {outpath}")
        # Print Summary
        logger.info("\n--- Eval Summary ---")
        success_df = results_df[results_df['Status'] == 'Success']
        if not success_df.empty:
            logger.info(f"Success: {len(success_df)}/{total}")
            for metric in ["Latency_ms", "Peak_Script_Memory_MB", "BLEU_Score", "ROUGE-L_F1", "BERTScore_F1"]:
                avg = success_df[metric].mean()
                display_metric = metric.replace('_', ' ').replace(' ms', '(ms)').replace(' mb', '(MB)')
                logger.info(f"Avg {display_metric}: {avg:.4f}")
        else: logger.warning("No successful queries.")
        logger.info("--------------------" )
    except Exception as e: logger.error(f"CSV save error: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG API.")
    parser.add_argument("--mode", type=str, required=True, choices=['hybrid', 'hybrid-rerank'], help="API mode to eval")
    args = parser.parse_args()
    run_eval(args.mode)
