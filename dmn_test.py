import torch
import torch.nn as nn

from module import QuestionModule, AnswerModule, InputModule
from memory import AttentionGRU, AttentionGRUCell, EpisodicMemory
from dynamic_memory_network_plus import DynamicMemoryNetworkPlus

# parameters for testing
VOCAB_SIZE = 100
EMBEDED_SIZE = 30
HIDDEN_SIZE = 4
CONTEXTS_LEN = 10
QUESTIONS_LEN = 5
ANSWERS_LEN = 3
BATCH_SIZE = 2


def test_attention_cell():
    attention_gru_cell = AttentionGRUCell(HIDDEN_SIZE, HIDDEN_SIZE)
    fact = torch.randn(2, HIDDEN_SIZE)
    c = torch.randn(2, HIDDEN_SIZE)
    g = torch.randn(2)
    c = attention_gru_cell.forward(fact, c, g)
    print("context size: {}".format(c.size()))


def test_attention_gru():
    input_size = HIDDEN_SIZE
    sentence_len = 4
    batch_size = 2
    attention_gru = AttentionGRU(input_size, HIDDEN_SIZE)
    facts = torch.randn(sentence_len, batch_size, input_size)
    G = torch.randn(batch_size, sentence_len)
    context = attention_gru.forward(facts, G)
    print("context size: {}".format(context.size()))
    print("context value: \n{}".format(context))


def text_episodic_memory():
    sentence_len = 4
    batch_size = 2
    episodic_memory = EpisodicMemory(HIDDEN_SIZE)
    facts = torch.randn(sentence_len, batch_size, HIDDEN_SIZE)
    questions = torch.randn(1, batch_size, HIDDEN_SIZE)
    memory = torch.randn(1, batch_size, HIDDEN_SIZE)
    episodic_memory.make_interaction(facts, questions, memory)
    # print("g size: {}".format(g.size()))
    # print("g value:\n {}".format(g))
    # memory = episodic_memory.forward(facts, questions, memory)
    # print("memory size: {}".format(memory.size()))


def test_question_module():

    question_module = QuestionModule(EMBEDED_SIZE, HIDDEN_SIZE)
    embedding = nn.Embedding(VOCAB_SIZE, EMBEDED_SIZE)
    questions = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 2]], dtype=torch.long)
    print('hidden size: {}'.format(HIDDEN_SIZE))
    print('input question size: {}'.format(questions.size()))
    questions = question_module.forward(questions, embedding)
    print('questions size: {}'.format(questions.size()))


def test_input_module():
    input_module = InputModule(EMBEDED_SIZE, HIDDEN_SIZE)
    embedding = nn.Embedding(VOCAB_SIZE, EMBEDED_SIZE)
    contexts = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 2]], dtype=torch.long)
    print("hidden_size: {}".format(HIDDEN_SIZE))
    print("context size: {}".format(contexts.size()))
    facts = input_module.forward(contexts, embedding)
    print("facts size: {}".format(facts.size()))


def test_answer_module():
    answer_module = AnswerModule(VOCAB_SIZE, EMBEDED_SIZE, HIDDEN_SIZE)
    word_embedding = nn.Embedding(VOCAB_SIZE, EMBEDED_SIZE)
    words = torch.zeros(2, 1, dtype=torch.long)
    print("words size: {}".format(words.size()))
    memory = torch.randn(1, 2, HIDDEN_SIZE)
    questions = torch.randn(1, 2, HIDDEN_SIZE)
    hidden = torch.cat([memory, questions], dim=2)
    words, hidden = answer_module.forward(hidden, words, word_embedding)
    # TOTAL_TURN = 5
    # for turn in range(TOTAL_TURN):
    #     # print("insize for")
    #     output, hidden = answer_module.forward(hidden, words, word_embedding)
    #     _, words = output.topk(1)
    #     words = words.long()
    print("words size: {}".format(words.size()))


def test_dmnp():
    model = DynamicMemoryNetworkPlus(VOCAB_SIZE, EMBEDED_SIZE, HIDDEN_SIZE)
    contexts = torch.randint(VOCAB_SIZE, size=(
        BATCH_SIZE, CONTEXTS_LEN), dtype=torch.long)
    questions = torch.randint(VOCAB_SIZE, size=(
        BATCH_SIZE, QUESTIONS_LEN), dtype=torch.long)
    preds = model.forward(contexts, questions, 3)
    print("preds size: {}".format(preds.size()))


def text_dmnp_loss():
    model = DynamicMemoryNetworkPlus(VOCAB_SIZE, EMBEDED_SIZE, HIDDEN_SIZE)
    contexts = torch.randint(VOCAB_SIZE, size=(
        BATCH_SIZE, CONTEXTS_LEN), dtype=torch.long)
    questions = torch.randint(VOCAB_SIZE, size=(
        BATCH_SIZE, QUESTIONS_LEN), dtype=torch.long)
    answers = torch.randint(VOCAB_SIZE, size=(
        BATCH_SIZE, ANSWERS_LEN), dtype=torch.long)
    loss = model.loss(contexts, questions, answers)
    print("loss : {}".format(loss))


def test_dmnp_predict():
    model = DynamicMemoryNetworkPlus(VOCAB_SIZE, EMBEDED_SIZE, HIDDEN_SIZE)
    contexts = torch.randint(VOCAB_SIZE, size=(
        BATCH_SIZE, CONTEXTS_LEN), dtype=torch.long)
    questions = torch.randint(VOCAB_SIZE, size=(
        BATCH_SIZE, QUESTIONS_LEN), dtype=torch.long)
    preds = model.predict(contexts, questions)
    print("preds:\n{}".format(preds))


if __name__ == '__main__':
    # input_model = InputModule(100, 30)
    # test_input_module()
    # test_answer_module()
    # test_dmnp()
    # text_episodic_memory()
    # text_dmnp_loss()
    # test_dmnp_predict()
    text_episodic_memory()
