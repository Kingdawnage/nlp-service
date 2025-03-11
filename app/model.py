import tf_keras as tfk
from transformers import TFBertModel, BertTokenizer

class ResumeAnalyzerModel(tfk.Model):
    def __init__(self, transformer, num_labels):
        super(ResumeAnalyzerModel, self).__init__()
        self.transformer = transformer
        self.token_classifier = tfk.layers.Dense(num_labels, activation='softmax')
        self.regressor = tfk.layers.Dense(1, activation='linear')
        
    def call(self, inputs, training=False):
        outputs = self.transformer(inputs)
        sequence_output = outputs.last_hidden_state
        token_logits = self.token_classifier(sequence_output)
        cls_output = sequence_output[:, 0, :]
        overall_score = self.regressor(cls_output)
        return token_logits, overall_score

def get_model(num_labels: int = 5):
    """
    Load the BERT model and tokenizer, then instantiate the ResumeAnalyzerModel.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transformer = TFBertModel.from_pretrained('bert-base-uncased')
    model = ResumeAnalyzerModel(transformer, num_labels)
    optimizer = tfk.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, 
                  loss=[tfk.losses.SparseCategoricalCrossentropy(), tfk.losses.MeanSquaredError()])
    return model, tokenizer