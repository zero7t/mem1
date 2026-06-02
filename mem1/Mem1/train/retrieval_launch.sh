
file_path=/root/paddlejob/workspace/new_mem1/mem1/mem1/assets/wiki-18
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/root/paddlejob/workspace/new_mem1/mem1/mem1/assets/models/intfloat__e5-base-v2

CUDA_VISIBLE_DEVICES=6,7 python rollout/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
