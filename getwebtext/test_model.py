
from cf import TTEModel as cf_model
from train_utils import load as cf_load
from train_utils import read_config
import torch
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

if __name__=="__main__":
    config = read_config("config/tte_small.yaml")
    model = cf_model(config)
    model = cf_load(model, config['saved_model_path'])
    model.eval()
    print('run model inference')
    kws= ['analytics']
    items = [
        'abbott',
        'abbvie',
        'aercap holdings n.v.',
        'aes corp.',
        'aethlon medical inc.',
        'agree realty corporation']
    with torch.no_grad():
        query_embed = model.query_model({config['query_col']: kws})
        query_embed = query_embed.detach().cpu().numpy()
        item_embed = model.item_model({config['ref_col']: items})
        item_embed = item_embed.detach().cpu().numpy()

    print(query_embed)

    print(item_embed)

    print()