# 외부 묘듈 불러오기
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
import tensorflow as tf
import pandas as pd
import tqdm

# 상수들은 따로 로컬 파일에서 불러오기
from config import *

# csv 형태의 훈련 데이터 읽기
# Read train data of csv format
train_data = pd.read_csv('ChatbotData.csv')

# 배치 사이즈와 훈련 데이터 길이에 따른 스텝 수 결정
# Determine the step number regard by the batch size and the total lenght of train data
steps = len(train_data) // BATCH_SIZE + 1

# KoGPT2 토크나이저, KoGPT2 모델 그리고 아담 옵티마이저 불러오기
# Call KoGPT2 as tokenizer and model, and call Adam as optimizers
koGPT2_TOKENIZER = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token=BOS, eos_token=EOS, pad_token=PAD)
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=EPSILION)

# 전체 훈련 데이터에서 문답, 라벨 추출을 위한 함수 정의
# Define the function to extract the quesion, answers and label from the total train data 
def get_chat_data():
    # 각 문장 최대 길이 설정
    # Set the maximum length the each sentences
    max_length = 44

    # Q 이하의 행을 질문으로, A 이하의 행을 답으로 간주하고 반복문으로 추출
    # Extract the Quesion from Q column and answeres from the A columns from A with loop
    for question, answer in zip(train_data.Q.to_list(), train_data.A.to_list()):

        # 문장 시작 및 종료 토큰으로 각 문장을 인코딩 후 결합 및 리턴
        # Combine and return the each sentence after encode with the sentence start mark and end mark
        bos_token = [koGPT2_TOKENIZER.bos_token_id]
        eos_token = [koGPT2_TOKENIZER.eos_token_id]
        sent = koGPT2_TOKENIZER.encode('' + question + '' + answer, max_length=max_length, truncation=True)
        yield bos_token + sent + eos_token

# 32비트 정수의 데이터형으로 텐서플로우 미지정 모양으로 위 함수에서 추출한 데이터를 넣고 변수로 지정
# Set the data shaped None and 32bit integer data type which extracted from the above function as "dataset" variable
dataset = tf.data.Dataset.from_generator(get_chat_data, output_signature=tf.TensorSpec(shape=[None], dtype=tf.int32))

# 명시적으로 텐서 형태의 미지정 모양으로 패딩할 모양을 설정
# Create shape None shaped of tensor explicity
padded_shapes = tf.TensorShape([None],)

# 배치 사이즈, 패딩 모양, 사용 토큰을 명시한 데이터셋으로 패딩
# Padding the dataset epxlict the batch size, padded shape and the padding value token
dataset = dataset.padded_batch(batch_size=BATCH_SIZE, padded_shapes=padded_shapes, padding_values=koGPT2_TOKENIZER.pad_token_id)

# 전체 배치 데이터셋 확인 (선택 사항)
# Check the dataset batch (Optional)
# for batch in dataset:
#     print(batch)
#     break

# 디코딩 후 첫번째 배치 출력값 확인 (선택 사항)
# Decode the first batch after the decode (Optional)
# koGPT2_TOKENIZER.decode(batch[0])
# batch[0]

# 에포치 수만큼 학습 수행
# Excute the training as many time as epoch
for epoch in range(EPOCHS):
    # 각 학습이 끝날 때마다 학습 손실률 초기화
    # Reset the epoch loss when every epoch traning is done
    epoch_loss = 0
    
    # 스텝 수와 패딩한 데이터 셋을 기준에 따라 학습을 수행하고 기존 어텐션 마스크를 토크나이저에 맞게 변경 
    # Strat the training follows the step number and the dataset. Change the default attention maks fits to the tokenizer
    for batch in tqdm.notebook.tqdm(dataset, total=steps):
        attention_mask =  tf.math.not_equal(batch, koGPT2_TOKENIZER.pad_token_id)
        
        with tf.GradientTape() as tape:
            # result = model(batch, labels=batch, training=True)

            # 이전에 정한 배치, 라벨, 어텐션 마스크를 적용한 모델에서 결과 값을 출력
            # Get the result from the model that applyed the preset batch, label,and attention mask 
            result = model(batch, labels=batch, attention_mask=attention_mask, training=True)
            # 결과에서 손실률을 측정
            # evaluate the loss from the result
            loss = result[0]
            # 훈련 배치 손실에서 평균값을 출력
            # Get the mean of the each training batch loss
            batch_loss = tf.reduce_mean(loss)

        grads = tape.gradient(batch_loss, model.trainable_variables)
        adam.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += batch_loss / steps

    print('Epoch {:>4} : loss_mean = {:>.9}'.format(epoch + 1, epoch_loss))

# 학습 종료 후 현재 경로에 모델 저장
# Save the model on current directory after finish the training
model.save('./')

def return_answer_by_chatbot(input_text):
    sent = '' + input_text + ''

    input_ids = [koGPT2_TOKENIZER.bos_token_id] + koGPT2_TOKENIZER.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])

    output = model.generate(input_ids, max_length = 44, early_stopping= True, eos_token_id=koGPT2_TOKENIZER.eos_token_id)
    sentence = koGPT2_TOKENIZER.decode(output[0].numpy().tolist())

    chatbot_response = sentence.split(' ')[1].replace('', '')

    return chatbot_response

return_answer_by_chatbot("안녕!")