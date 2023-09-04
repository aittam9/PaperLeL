import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict, Counter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import SparsePCA 
import joblib 
import torch 


"""Utils module containing all the functions used for the experiments."""

# function to estimate corrs 
def estimate_corrs(y_original, y_inferred, labels, corr_type):
    """
    corr_type è un parametro che specifica se la correlazione deve essere 
    calcolata tra le righe ("byrow") o tra le colonne ("bycolumn") delle due
    matrici fornite in input
    
    In out fornisce sia la correlazione media che un Counter con le correlazioni puntuali
    """
    corrs = Counter()
        
    if corr_type == "byrow":
        for i in range(len(y_original)):
            rho, _ = spearmanr(y_original[i], y_inferred[i])
            corrs[labels[i]] = rho
        
    elif corr_type == "bycolumn":
        for i in range(y_original.shape[1]):
            rho, _ = spearmanr(y_original[:,i], y_inferred[:,i])
            corrs[labels[i]] = rho
                
    average_rho = np.average(list(corrs.values()))
        
    return corrs, average_rho

#helper function to load the embeddings and reshape them along with the sspace
def embeddings_preparation(model_emb_path, 
                           sspace_nsubj , 
                           sspace_dobj, model_name = "", 
                           return_new_sspace = False,
                           dir_path = ""):
    
    """Parameters:
    model_emb_path: name under wich embeddings are saved
    sspace_nsubj: the sspace loaded for the nsubj
    sspace_dobj: the sspace loaded for the dobj
    model_name: the name of the model we are referringe to. Will be used to store the spaces with the right key.
    return_new_sspace: parameter to decide if we want in output the reshaped sspaces. To be used the first time only.
    
    Output:
    a dictionary containing two keys with respective values: the actual embedding space and a baseline with the same shape
    new_sspace_nsubj, new_sspace_dobj: the reshaped target semantic spaces (only if param return_new_sspace =True)
    """
    
    #load the embeddings for the given model
    with open(os.path.join(dir_path,model_emb_path), "rb") as infile:
        embeddings_dict = pickle.load(infile)
    
    #convert the dictionary into a dataframe
    emb_df = pd.DataFrame(embeddings_dict).T 
    
    #take the index of the semantic spaces to match the extracted verbs
    verbs2keep_nsubj = sspace_nsubj.index.tolist()
    verbs2keep_dobj = sspace_dobj.index.tolist()
    
    #filter the verbs from the target semantic space to match the embeddings spaces
    model_nsubj = emb_df.filter(items = verbs2keep_nsubj, axis = 0).apply(lambda x:(x-x.min())/(x.max()-x.min()))
    model_dobj = emb_df.filter(items = verbs2keep_dobj, axis = 0 ).apply(lambda x:(x-x.min())/(x.max()-x.min()))

    #reverse the process to restrict the semantic spaces
    sspace_nsubj_final = sspace_nsubj.filter(model_nsubj.index.tolist(), axis = 0)
    sspace_dobbj_final = sspace_dobj.filter(model_dobj.index.tolist(), axis = 0)
    
    #store all the spaces along with a baseline in a dictionary
    spaces_dict_nsubj = {}
    spaces_dict_nsubj[model_name+"_nsubj"] = model_nsubj.values 
    spaces_dict_nsubj[model_name+"_nsubj_baseline"] = np.random.random_sample(model_nsubj.shape)
    
    # same for dobj
    spaces_dict_dobj = {}
    spaces_dict_dobj[model_name+"_dobj"] = model_dobj.values 
    spaces_dict_dobj[model_name+"_dobj_baseline"] = np.random.random_sample(model_dobj.shape)
    
    if return_new_sspace:
        return spaces_dict_nsubj, spaces_dict_dobj, sspace_nsubj_final, sspace_dobbj_final
    else:
        return spaces_dict_nsubj, spaces_dict_dobj


#helper function to automate correlations estimation
def get_correlations_values(model_space:dict, semantic_space: pd.DataFrame, sPca = True):
    
    """Parameters:
    model_space: a dictionary containig 2 keys: the actual model embedding space for the gram argument(e.g. nsubj)
                 and the random baseline shape as the model space.
    semantic_space: a data frame the target semantic space for the referred gram argument. Will be used as y and to derive labels
                    for corrs estimation.
    sPCA: a boolean to decide if we want to transform the target space with sPCA. Default to True to reduce noise.
    
    Output:
    Print average rhos values, by column and by row, for the embedding space and the baseline
    all_corrs: puncutal correlations by row and column for model and baseline.
    all_avg_rhos: average rhos by row and column, for model and baseline."""
    
    # initialize the regressor and dictionary to be filled
    pls = PLSRegression(n_components= 10)
    all_corr = defaultdict(dict)
    all_avg_rhos = defaultdict(dict)
    y_pred = {}
    #get the verbs and properties names, to be used as labels in corr estimation
    verbs = semantic_space.index.tolist()
    properties = semantic_space.columns.tolist()
    
    #initialize the sPCA and transform the target space if param not False
    if sPca:
        print(f"Sparse PCA activated\n")
        pca = SparsePCA(n_components=14)
        y = pca.fit_transform(semantic_space.values)
    else:
        print(f"Not using Sparse PCA")
        y = semantic_space.values
        
    # mapping for the given space
    for k in model_space.keys():
        X = model_space[k]
        y = y
        #get prediction values with cross validation k=10
        y_pred[k] = cross_val_predict(pls, X, y, cv = 10)

        # store and print correlation by row    
        corrs, avg_rho = estimate_corrs(y,y_pred[k], verbs, 'byrow')
        print(f"Average row correlation for the {k} space: {avg_rho}")

        all_corr[k]['byrow'] = corrs
        all_avg_rhos[k]['byrow'] = avg_rho

        # store and print correlation by column
        corrs, avg_rho = estimate_corrs(y,y_pred[k], properties, 'bycolumn')
        print(f"Average column correlation for the {k} space: {avg_rho}\n")

        all_corr[k]['bycolumn'] = corrs
        all_avg_rhos[k]['bycolumn'] = avg_rho

    return all_corr, all_avg_rhos 

#helper function to aggregate all correlation results in a unique dataframe
def aggregate_results(list_of_dict, save_results = False, file_name = "", output_path = ""):
    outdir = output_path
    df = pd.concat([pd.DataFrame(i).T for i in list_of_dict])
    
    if save_results:
        file_name = file_name+".csv"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        df.to_csv(os.path.join(outdir,file_name), sep = "\t")
        
        return df
    else:
        return df

"""Set of functions to combine subwords, subembeddings, extract feature and target representations"""

#word pieces combination based on tokens ids
def combine_subwords(ids, words):
    
    """The function takes in two arguments both obtained with hf TokenizerFast class:\
    ids: a list of successive non-single ids \
    word_pieces: a list of word pieces
    
    output:
    a dictionary mapping the ids to their respective subwords
    a list of the reconstructed words"""
    
    words = list(map(lambda x:x.replace("Ġ", "").strip(), words))
    
    # Ensure both input lists have the same length
    if len(ids) != len(words):
        raise ValueError("Input lists must have the same length")

    # Create a dictionary to store word pieces by id
    id_word_dict = {}

    # Iterate through the lists and populate the dictionary
    for id, word in zip(ids, words):
        if id not in id_word_dict:
            id_word_dict[id] = []
        id_word_dict[id].append(word)

    # Create the list of tuples by joining the word pieces
    result = [(id, ''.join(word_pieces)) for id, word_pieces in id_word_dict.items() if not id == None]

    #get rid of None key if any. If present is for the special tokens /s\s
    if id_word_dict.get(None):
      del id_word_dict[None]

    return id_word_dict, result

#word pieces embedding combination based on tokens ids
def combine_subembeddings(ids, embeddings, device = None):

    # Ensure both input lists have the same length
    if len(ids) != len(embeddings):
        raise ValueError("Input lists must have the same length")

    # Create a dictionary to store embedding of word pieces by id
    id_emb_dict = {}

    # Iterate through the lists and populate the dictionary
    for id, sub_emb in zip(ids, embeddings):
        if id not in id_emb_dict:
            id_emb_dict[id] = []
        if device:
          id_emb_dict[id].append(sub_emb.cpu().numpy().astype(float))

        else:
          id_emb_dict[id].append(sub_emb.numpy().astype(float))

    # Create the list of tuples by averaging embedding pieces
    result = [(id, np.mean(sub_emb, axis = 0)) for id, sub_emb in id_emb_dict.items() if not id == None]

    #get rid of None key if any. If present is for the special tokens /s\s
    if id_emb_dict.get(None):
      del id_emb_dict[None]

    return id_emb_dict, result

# helper function to extract representations with a given model
def feature_extractor(sent, token, tokenizer, model, device = None):    #token
    tokenized_sent = tokenizer(sent, return_tensors = "pt", truncation = True)
    word_ids = tokenized_sent.word_ids()
    #dynamically get the target token id
    _, combined_words = combine_subwords(word_ids, tokenized_sent.tokens())
    combined_words = [i[1] for i in combined_words]
    #ensure to get both lower cased and non-lower cased tokens (different between tokenizers)
    try:
      tokid = combined_words.index(token.lower())
    except:
      tokid = combined_words.index(token)
    
    #insert code for the gpu
    if device:
      with torch.no_grad():
        output = model(**tokenized_sent.to(device))
    else:
      with torch.no_grad():
          output = model(**tokenized_sent)
    embeddings = output["last_hidden_state"][0,:]
    embs_dict, encoded_sent_fw = combine_subembeddings(word_ids, embeddings, device = device)
    
    return  embs_dict, encoded_sent_fw, tokid

#helper function to select the target embeddings
def extract_target_embs(encoded_sent_fw, tokid, embs_dict):
    target= encoded_sent_fw[tokid][1]
    target_sub_embs = embs_dict[tokid]

    return target, target_sub_embs

#main function to loop over all the sentences and get the target representations
def get_target_embeddings(sents, tokens, sent_ids, lemmas, tokenizer, model, device = None):  
    if device:
      device = device
    target_embeddings = {}
    total_sub_embs = {}
    #loop over the sentences to extract each representation
    for i in tqdm(range(len(sents))):
        
        sent_id = str(sent_ids[i])
        token = tokens[i]
        sent = sents[i]
        lemma = lemmas[i]
        #extract the features for the whole sentence
        embs_dict, encoded_sent_fw, target_tokid = feature_extractor(sent, token, tokenizer, model, device =device)   #token
    
        #extract the target embeddings from the given sentence
        target, target_sub_embs = extract_target_embs(encoded_sent_fw, target_tokid, embs_dict)
        #join the token and sent id to create a key for the dict
        key = lemma +"."+sent_id
        #add value to the key
        target_embeddings[key] = target
        #store the sub embs in a dictionary with k=word.semt_id: [e1...en]
        total_sub_embs[key] = target_sub_embs
  
    return target_embeddings, total_sub_embs

#helper func to write the results
def serialize_embs(embs, file_name:str, model_ckp:str, output_path = ""):
    #write the output in the dedicated directory
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if "/" in model_ckp:
        model_ckp = model_ckp.split("/")[-1]

    with open(os.path.join(output_path, file_name)+model_ckp.split("/")[-1]+".pkl", "wb") as outfile:
        pickle.dump(embs, outfile)
    return print("Done")

#function to train the regressors and save them into a given directory
def train_regressor(model_spaces:[dict,pd.DataFrame],target_sspace:pd.DataFrame, sPca = True,
                     save_model = False, model_name = "", output_path = ""):
    """Params:
    model_spaces: dicitonary containig the embeddings and baseline spaces.
    target_sspace: the target nsubj sspace.
    save_model: Boolean to control model saving.
    model_name: empty string to be filled if model has to be saved
    
    Output:
    regr: fitted regressor
    saved model (if save_model = True)"""
    
    pls = PLSRegression(n_components=10)
    
    if type(model_spaces) == pd.core.frame.DataFrame:
        X = model_spaces.values
    
    elif type(model_spaces) == dict:
        X = np.array(list(model_spaces.values())[0])
    else:
        print("Invalid model space data type\nSpace shoud be dictionary o pandas dataframe")
    
    #transform y with sPca or not
    if sPca:
        print(f"Sparse PCA activated\n")
        pca = SparsePCA(n_components=14)
        y = pca.fit_transform(target_sspace.values)
    else:
        print(f"Not using Sparse PCA")
        y = target_sspace.values 

    regr = pls.fit(X,y)
    
    if save_model:
        filename = "pls_regr_"+model_name
        outdir = output_path
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        with open(os.path.join(outdir, filename+".pkl"), "wb") as outfile:
            joblib.dump(regr, outfile)
    
    return regr

