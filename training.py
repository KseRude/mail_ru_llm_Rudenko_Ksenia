from stat_lm import StatLM, Tokenizer
from datasets import load_dataset
import model_wrapper


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


def test_model():
    model = model_wrapper.ModelWrapper()
    status, result = model.load('StatLM')
    if not status:
        print(result)
        return
    input_text = 'поп пришел к бабе'
    status, result = model.generate(input_text)
    print(result)


if __name__ == '__main__':
    #train_model()
    test_model()
