from stat_lm import StatLM, Tokenizer
from datasets import load_dataset


def train_model():
    dataset = load_dataset("IgorVolochay/russian_jokes")
    train_texts = [el['text'] for el in dataset['train'].to_list()]

    tokenizer = Tokenizer()
    tokenizer.build_vocab(train_texts)
    tokenizer_path = 'models/stat_lm/tokenizer.pkl'
    tokenizer.save(tokenizer_path)
    stat_lm_model = StatLM(tokenizer)
    stat_lm_model.train(train_texts)
    stat_lm_path = 'models/stat_lm/stat_lm.pkl'
    stat_lm_model.save_stat(stat_lm_path)


if __name__ == '__main__':
    train_model()