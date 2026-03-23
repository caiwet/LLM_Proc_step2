import pyarrow.parquet as pq

def show_meta(path):
    pf = pq.ParquetFile(path)
    md = pf.schema_arrow.metadata
    print("\n===", path, "===")
    print("has metadata:", md is not None)
    if md:
        print("metadata keys:", list(md.keys()))
        if b"huggingface" in md:
            # print only a small prefix
            blob = md[b"huggingface"]
            print("huggingface metadata prefix:", blob[:300])

show_meta("/data/bwh-comppath-img2/MGH_CID/LLM_Proc/step2_gemini_output/eval_tokenized_lfm.parquet")
show_meta("/data/bwh-comppath-img/concept_extraction/data/concept_extraction/concept_extraction_500k_eval_tokenized_lfm.parquet")