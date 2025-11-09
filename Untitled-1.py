with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()


def len_of_text():
	print("Total number of character:", len(raw_text))
	print(raw_text[:99])


import re


def ways_to_preprocess():
	text = "Now, I am currently worrying about whether I can make pregress."
	result = re.split(r'(\s)', text)  #\s means splitting any white spaces
	#['Now,', ' ', 'I', ' ', 'am', ' ', 'currently', ' ', 'worrying', ' ', 'about', ' ', 'whether', ' ', 'I', ' ', 'can', ' ', 'make', ' ', 'pregress.']
	print(result)

	result = re.split(r'([,.]|\s)', text)
	#['Now', ',', '', ' ', 'I', ' ', 'am', ' ', 'currently', ' ', 'worrying', ' ', 'about', ' ', 'whether', ' ', 'I', ' ', 'can', ' ', 'make', ' ', 'pregress', '.', '']
	print(result)

	result = [item for item in result if item.strip()]
	print(result)  #removed white text
	#['Now', ',', 'I', 'am', 'currently', 'worrying', 'about', 'whether', 'I', 'can', 'make', 'pregress', '.']

	text = "Now, I am currently worrying-- about whether I can make pregress?"
	result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
	result = [item.strip() for item in result if item.strip()]
	print(result)
	#['Now', ',', 'I', 'am', 'currently', 'worrying', '--', 'about', 'whether', 'I', 'can', 'make', 'pregress', '?']


def preprocess_and_tokenize(v):
	preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
	preprocessed = [item.strip() for item in preprocessed if item.strip()]

	def test1():  #print(preprocessed[:30])
		print(preprocessed[:30])

	all_words = sorted(set(preprocessed))
	all_words.extend(["<|endoftext|>", "<|unk|>"])

	def test2():  #vocabulary size
		vocab_size = len(all_words)
		print(f"Vocabulary size: {vocab_size}")

	vocab = {token: integer for integer, token in enumerate(all_words)}

	def test3():  #enumerate
		for i, item in enumerate(vocab.items()):
			print(item)
			if i >= 50: break

	def tokenize_all():
		tokenizer = SimpleTokenizerV1(vocab)
		ids = tokenizer.encode(raw_text)
		print(ids)

		decoded_text = tokenizer.decode(ids)
		print(decoded_text)

	def self_experimentsV1():
		tokenizer = SimpleTokenizerV1(vocab)
		#text="Hello, world?" Hello is not in the vocabulary.
		#print(tokenizer.encode(text)) This will give an KeyError
		text = "What, is it?"
		print(tokenizer.encode(text))  #[109, 5, 584, 585, 10]
		print(tokenizer.decode(tokenizer.encode(text)))  #What, is it?

	def self_experimentsV2():
		text1 = "Hello, do you like tea?"
		text2 = "In the sunlit terraces of the palace."
		text = " <|endoftext|> ".join((text1, text2))
		print(text)
		tokenizer = SimpleTokenizerV2(vocab)
		print(tokenizer.encode(text))
		print(tokenizer.decode(tokenizer.encode(text)))

	if v == 'all':
		tokenize_all()
		print("\nOutput is tokenize_all()")
	if v == 'expv1':
		self_experimentsV1()
		print("\nOutput is self_experiments()")
	if v == 'test':
		test1()
		test2()
		test3()
		print("\nOutput is test()")
	if v == 'expv2':
		self_experimentsV2()
		print("\nOutput is self_experimentsV2()")


class SimpleTokenizerV1:

	def __init__(self, vocab):
		self.str_to_int = vocab
		#将传入的词表字典赋值给实例变量 self.str_to_int，用来从字符串查找对应的整数ID
		self.int_to_str = {i: s for s, i in vocab.items()}
		#创建一个反向字典 self.int_to_str，用于从整数ID查找对应的字符串。这里使用字典推导式遍历 vocab，交换键和值。

	def encode(self, text):
		preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
		'''
		使用正则表达式 re.split 对字符串 text 进行分割，分割规则是匹配以下内容：
		逗号、点、冒号、分号、问号、下划线、感叹号、双引号、括号、单引号
		连续两个减号 --
		任意空白字符 \s
		re.split 的特点是捕获组内的分隔符也会被保留在结果列表中，所以这里分割后结果包含单词和标点符号。
		'''
		preprocessed = [item.strip() for item in preprocessed if item.strip()]
		'''
		对分割结果进行处理：
		用 strip() 去除每个元素首尾的空白字符；
		过滤掉空字符串，保留非空的元素。
		这样得到一个干净的token列表，包括单词和标点。
		'''
		ids = [self.str_to_int[s] for s in preprocessed]
		#将每个分词后的token用词表 self.str_to_int 查找对应的整数ID，生成一个整数ID列表 ids。
		return ids

	def decode(self, ids):
		text = " ".join(self.int_to_str[i] for i in ids)
		'''
		先通过 self.int_to_str 把每个ID转换回对应的字符串token，然后用空格 " " 把它们连接成一个字符串 text。
		此时，所有token之间有空格，标点符号前也有空格。
		'''
		#Replace spaces before the specified punctuations
		text = re.sub(r'\s+([,.:;?_!"()\']|--)', r'\1', text)
		'''
		通过正则表达式 re.sub 处理字符串：
		匹配所有指定的标点符号（和编码时一样的集合）及空白字符；
		用匹配到的符号本身替换它（即删除符号前面的空格）。
		这一步是为了去除标点符号前多余的空格，使文本格式更符合自然语言的书写习惯。

		'''
		return text


class SimpleTokenizerV2:

	def __init__(self, vocab):
		self.str_to_int = vocab
		self.int_to_str = {i: s for s, i in vocab.items()}

	def encode(self, text):
		preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
		preprocessed = [item.strip() for item in preprocessed if item.strip()]
		preprocessed = [  #Change
				item if item in self.str_to_int else "<|unk|>"
				for item in preprocessed
		]
		ids = [self.str_to_int[s] for s in preprocessed]
		return ids

	def decode(self, ids):
		text = " ".join(self.int_to_str[i] for i in ids)
		text = re.sub(r'\s+([,.:;?_!"()\']|--)', r'\1', text)
		return text


#preprocess_and_tokenize('expv2')


def tiktoken():
	import importlib, tiktoken
	tokenizer = tiktoken.get_encoding("gpt2")
	integers = tokenizer.encode("Akwirw ier")
	print(integers)
	strings = tokenizer.decode(integers)
	print(strings)
	#[33901, 86, 343, 86, 220, 959]
	#Akwirw ier

	enc_text = tokenizer.encode(raw_text)
	print(len(enc_text))  #5145
	enc_sample = enc_text[50:]

	context_size = 4
	x = enc_sample[:context_size]
	y = enc_sample[1:context_size]
	print(f"x: {x}")
	print(f"y:      {y}")
	#x: [290, 4920, 2241, 287]
	#y:      [4920, 2241, 287]

	for i in range(1, context_size + 1):
		context = enc_sample[:i]
		desired = enc_sample[i]

		print(context, "---->", desired)
		print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
		'''
		[290] ----> 4920
		and ---->  established
		[290, 4920] ----> 2241
		and established ---->  himself
		[290, 4920, 2241] ----> 287
		and established himself ---->  in
		[290, 4920, 2241, 287] ----> 257
		and established himself in ---->  a '''


import torch, tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDataSetV1(Dataset):

	def __init__(self, txt, tokenizer, max_length, stride):
		self.input_ids = []
		self.target_ids = []

		#Tokenize the entire text
		token_ids = tokenizer.encode(txt,
																 allowed_special=set("<|endoftext|>", ))

		#Use a sliding window to chunk the book into overlapping sequences of max_length
		for i in range(0, len(token_ids) - max_length, stride):
			input_chunk = token_ids[i:i + max_length]
			target_chunk = token_ids[i + 1:i + max_length + 1]
			self.input_ids.append(torch.tensor(input_chunk))
			self.target_ids.append(torch.tensor(target_chunk))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, ids):
		return self.input_ids[ids], self.target_ids[ids]


def create_dataloader_v1(
		txt,
		batch_size=4,
		max_length=256,
		stride=128,
		shuffle=True,
		drop_last=True,
		#drop_last = true drops the last batch if it is shorter than specified batch_size to prevent lost spikes during training
		num_workers=0):
	#Initialize the tokenizer
	tokenizer = tiktoken.get_encoding("gpt2")

	#Create dataset
	dataset = GPTDataSetV1(txt, tokenizer, max_length, stride)

	#Create dataloader
	dataloader = DataLoader(dataset,
													batch_size=batch_size,
													shuffle=shuffle,
													drop_last=drop_last,
													num_workers=num_workers)

	return dataloader


def access_dataloader():
	dataloader = create_dataloader_v1(raw_text,
																		batch_size=1,
																		max_length=4,
																		stride=1,
																		shuffle=False)

	data_iter = iter(dataloader)
	first_batch = next(data_iter)
	print(first_batch)

	dataloader = create_dataloader_v1(raw_text,
																		batch_size=1,
																		max_length=4,
																		stride=4,
																		shuffle=False)
	data_iter = iter(dataloader)
	inputs, targets = next(data_iter)
	print("Inputs:\n", inputs)
	print("\nTargets:\n", targets)
	'''
	[tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
	Inputs:
	tensor([[  40,  367, 2885, 1464]])

	Targets:
	tensor([[ 367, 2885, 1464, 1807]])
	'''


import gensim.downloader as api
import numpy as np
from gensim.models import KeyedVectors


def vector_embeddings_handson_example():
	model = api.load(
			"word2vec-google-news-300")  #Download and return as onject when ready
	word_vectors = model
	print(word_vectors['computer']
				)  #Accessing the vector for the word 'computer'
	print(word_vectors['cat'].shape)  #(300,)

	#Example of using most_similar: King + Woman - Man = ?
	print(
			word_vectors.most_similar(positive=['king', 'woman'],
																nagative=['man'],
																topn=10))

	print(word_vectors.similarity('woman', 'man'))  #0.76....
	print(word_vectors.similarity('king', 'queen'))  #.65.....
	print(word_vectors.similarity('uncle', 'aunt'))
	print(word_vectors.similarity('boy', 'girl'))
	print(word_vectors.similarity('nephew', 'niece'))
	print(word_vectors.similarity('paper', 'water'))  #0.11......
	#Most common words
	print(word_vectors.similarity('tower', topn=5))

	#Words to compare
	word1 = 'man'
	word2 = 'woman'
	word3 = 'semiconductor'
	word4 = 'earthworm'
	word5 = 'nephew'
	word6 = 'niece'

	#Calculate the vector difference
	vector_difference1 = model[word1] - model[word2]
	vector_difference2 = model[word3] - model[word4]
	vector_difference3 = model[word5] - model[word6]

	#Calculate the magnitude of the vector difference
	magnitude_of_difference1 = np.linalg.norm(vector_difference1)  #1.73
	magnitude_of_difference2 = np.linalg.norm(vector_difference2)  #5.67
	magnitude_of_difference3 = np.linalg.norm(vector_difference3)  #1.96
	print(magnitude_of_difference1, magnitude_of_difference2,
				magnitude_of_difference3)


def vector_embeddings():
	#In this example: quick(4) fox(0) is(3) in(2) the(5) house(1)
	input_ids = torch.tensor([2, 3, 5, 1])
	vocab_size = 6
	output_dim = 3  #output dimention = 3

	torch.manual_seed(123)
	embedding_layers = torch.nn.Embedding(vocab_size, output_dim)
	#Randomly initialze the weights of the embedding Matrix

	print(embedding_layers.weight)  #Print weights
	'''
	Parameter containing:
	tensor([[ 0.3374, -0.1778, -0.1690],
			[ 0.9178,  1.5810,  1.3010],
			[ 1.2753, -0.2010, -0.1606],
			[-0.4015,  0.9666, -1.1481],
			[-1.1589,  0.3255, -0.6315],
			[-2.8400, -0.7849, -1.4096]], requires_grad=True)
	'''#These are to be optimized

	print(embedding_layers(torch.tensor([3])))  #[] is optional
	#tensor([-0.4015,  0.9666, -1.1481], grad_fn=<EmbeddingBackward0>)
	print(embedding_layers(input_ids))
	'''
	tensor([[ 1.2753, -0.2010, -0.1606],
			[-0.4015,  0.9666, -1.1481],
			[-2.8400, -0.7849, -1.4096],
			[ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
	'''


def positional_embeddings():
	vocab_size = 50257
	output_dim = 256
	#One embedding layer of 256 length is generated for each token in input.

	token_embedding_layer = torch.nn.Embedding(
			vocab_size, output_dim)  #Randomly generate value

	max_length = 4
	dataloader = create_dataloader_v1(raw_text,
																		batch_size=8,
																		max_length=max_length,
																		stride=max_length,
																		shuffle=False)
	data_iter = iter(dataloader)
	inputs, targets = next(data_iter)

	print("Token IDs:\n", inputs)
	'''
	Token IDs:
	tensor([[   40,   367,  2885,  1464],
			[ 1807,  3619,   402,   271],
			[10899,  2138,   257,  7026],
			[15632,   438,  2016,   257],
			[  922,  5891,  1576,   438],
			[  568,   340,   373,   645],
			[ 1049,  5975,   284,   502],
			[  284,  3285,   326,    11]])
	'''
	print("\nInputs shape:\n", inputs.shape)
	'''
	Inputs shape:
	torch.Size([8, 4])
	'''
	#Create token embeddings
	token_embeddings = token_embedding_layer(inputs)
	print(token_embeddings.shape)  #torch.Size([8, 4, 256])
	#Create positional embeddings
	context_length = max_length
	pos_embedding_layer = torch.nn.Embedding(
			context_length, output_dim)  #Randomly generate value
	#torch.Size([4, 256])
	#We only need 4 positional embeddings for each position in total.

	#Add the token embeddings + position embeddings (8*4*256)


inputs = torch.tensor([
	[0.43, 0.15, 0.89],  #Your		(x^1)
	[0.55, 0.87, 0.66],  #journey    (x^2)
	[0.57, 0.85, 0.64],  #starts    	(x^3)
	[0.22, 0.58, 0.33],  #with    	(x^4)
	[0.77, 0.25, 0.10],  #one		(x^5)
	[0.05, 0.80, 0.55]  #step		(x^6)
])


def simplified_attention_mechanism():
	#Attention scores
	query = inputs[1]  #2nd input token is query

	attn_scores_2 = torch.empty(inputs.shape[0])
	for i, x_i in enumerate(inputs):
		attn_scores_2[i] = torch.dot(x_i, query)  #Dot product

	print(attn_scores_2
				)  #tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]) Same with better method in later code

	#Normalization
	attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

	print("Attention weights:", attn_weights_2_tmp)
	print("Sum:", attn_weights_2_tmp.sum())

	#Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
	#Sum: tensor(1.0000)

	def softmax_naive(x):
		return torch.exp(x) / torch.exp(x).sum(dim=0)

	attn_weights_2_naive = softmax_naive(attn_scores_2)

	print("Attention weights:", attn_weights_2_naive)
	print("Sum:", attn_weights_2_naive.sum())
	#Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
	#Sum: tensor(1.)

	#PyTorch softmax (recommended)
	attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
	print("Attnetion weights:", attn_weights_2)
	print("sum:", attn_weights_2.sum())
	#Attnetion weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
	#sum: tensor(1.)

	#Calculation of context vectors:
	query = inputs[1]  #2nd input token is the query

	context_vec_2 = torch.zeros(query.shape)
	for i, x_i in enumerate(inputs):
		context_vec_2 += attn_weights_2[i] * x_i

	print(context_vec_2)  #tensor([0.4419, 0.6515, 0.5683]) Same with better method in later code

	#Calculation of attention score matrix for all queries
	attn_scores = torch.empty(6, 6)
	for i, x_i in enumerate(inputs):
		for j, x_j in enumerate(inputs):
			attn_scores[i, j] = torch.dot(x_i, x_j)

	print(attn_scores)
	'''
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
		[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865], see previous code when query = [1]
		[0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
		[0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
		[0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
		[0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
	'''
	#'For' loops are generally slow, and we can achieve the same results using matrix multiplication **BEST PRACTICE**
	attn_scores = inputs @ inputs.T  #Transpose
	print(attn_scores)

	#Normalization
	attn_weights = torch.softmax(attn_scores, dim=-1)#-1 will normalize the columns so that the values of the column dimension is equal to 1
	print(attn_weights)
	'''
tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
		[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
		[0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
		[0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
		[0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
		[0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
	'''
	
	#Verify row #2 equals to 1
	row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
	print("Row 2 sum:", row_2_sum)
	print("All row sums:", attn_weights.sum(dim=-1))
	#Row 2 sum: 1.0
	#All row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

	#Use these attention weights to compute all contest vectors via matrix multiplication.
	all_context_vecs = attn_weights @ inputs
	print(all_context_vecs)
	'''
tensor([[0.4421, 0.5931, 0.5790],
		[0.4419, 0.6515, 0.5683], exact the same as previous implementation
		[0.4431, 0.6496, 0.5671],
		[0.4304, 0.6298, 0.5510],
		[0.4671, 0.5910, 0.5266],
		[0.4177, 0.6503, 0.5645]])
	'''
import torch.nn as nn
def self_attention_mechanism():
	'''Converting input embeddings into key, query, and value vectors'''
	x_2 = inputs[1]#A the 2nd input element
	d_in = inputs.shape[1]#B the input embedding size, d=3
	d_out = 2#C The output embedding size, d_out=2

	torch.manual_seed(123)
	W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
	W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
	W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
	print(f"{W_query}\n{W_key}\n{W_value}")
	'''
	Parameter containing:
	tensor([[0.2961, 0.5166],
					[0.2517, 0.6886],
					[0.0740, 0.8665]])
	Parameter containing:
	tensor([[0.1366, 0.1025],
					[0.1841, 0.7264],
					[0.3153, 0.6871]])
	Parameter containing:
	tensor([[0.0756, 0.1966],
					[0.3164, 0.4017],
					[0.1186, 0.8274]])
	'''
	#Multiplication
	query_2 = x_2 @ W_query
	key_2 = x_2 @ W_key
	value_2 = x_2 @ W_value
	print(query_2)#tensor([0.4306, 1.4551])

	#Obtain all keys and values via matrix multiplication
	queries = inputs @ W_query
	keys = inputs @ W_key
	values = inputs @ W_value
	#Shapes all: torch.size([6, 2])

	attn_scores_2=query_2@keys.T
	print(attn_scores_2)
	#tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
	
	attn_scores = queries @ keys.T #omega
	print(attn_scores)
	'''
tensor([[0.9231, 1.3545, 1.3241, 0.7910, 0.4032, 1.1330],
				[1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440],
				[1.2544, 1.8284, 1.7877, 1.0654, 0.5508, 1.5238],
				[0.6973, 1.0167, 0.9941, 0.5925, 0.3061, 0.8475],
				[0.6114, 0.8819, 0.8626, 0.5121, 0.2707, 0.7307],
				[0.8995, 1.3165, 1.2871, 0.7682, 0.3937, 1.0996]])
	'''

	#Normalize them!
	#Before, the matrix have to be scaled by sqrt(d_keys)
	#Then softmax

	d_k=keys.shape[-1]#Column, so-1
	attn_weights_2=torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)#All, so-1
	print(attn_weights_2)#tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
	print(d_k)#2

	#Calculating context vectors
	context_vec_2=attn_weights_2@values
	print(context_vec_2)#tensor([0.3061, 0.8210])



		#compact self attention python class
	class SelfAttention_v1(nn.Module):
		def __init__(self,d_in,d_out):
			super().__init__()
			self.W_query	=	nn.Parameter(torch.rand(d_in, d_out))
			self.W_key		=	nn.Parameter(torch.rand(d_in, d_out))
			self.W_value	=	nn.Parameter(torch.rand(d_in, d_out))

		def forward(self,x):
			keys = 	x@self.W_key
			queries=x@self.W_query
			values=	x@self.W_value

			attn_scores = queries@keys.T
			attn_weights = torch.softmax(
				attn_scores/keys.shape[-1]**0.5, dim=-1
			)
	
			context_vec= attn_weights @ values
			return context_vec
	
	torch.manual_seed(123)
	sa_v1 = SelfAttention_v1(d_in, d_out)
	print(sa_v1(inputs))
	'''
tensor([[0.2996, 0.8053],
				[0.3061, 0.8210],
				[0.3058, 0.8203],
				[0.2948, 0.7939],
				[0.2927, 0.7891],
				[0.2990, 0.8040]], grad_fn=<MmBackward0>)
	'''
	class SelfAttention_v2(nn.Module):
		def __init__(self,d_in,d_out,qkv_bias=False):
			super().__init__()
			self.W_query	=	nn.Linear(d_in, d_out, bias=qkv_bias)
			self.W_key		=	nn.Linear(d_in, d_out, bias=qkv_bias)
			self.W_value	=	nn.Linear(d_in, d_out, bias=qkv_bias)
	
		def forward(self,x):
			keys = 	self.W_key(x)
			queries=self.W_key(x)
			values=	self.W_key(x)
	
			attn_scores = queries@keys.T
			attn_weights = torch.softmax(
				attn_scores/keys.shape[-1]**0.5, dim=-1
			)
	
			context_vec= attn_weights @ values
			return context_vec
	torch.manual_seed(789)
	sa_v2 = SelfAttention_v2(d_in, d_out)
	print(sa_v2(inputs))
	'''
	tensor([[ 0.0293, -0.3351],
		[ 0.0205, -0.3395],
		[ 0.0210, -0.3390],
		[ 0.0204, -0.3366],
		[ 0.0317, -0.3258],
		[ 0.0154, -0.3422]], grad_fn=<MmBackward0>)
	'''


	#Causal Attention:
	queries = sa_v2.W_query(inputs)
	keys = sa_v2.W_key(inputs)
	attn_scores = queries @ keys.T
	attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
	print(attn_weights)
	'''
	tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
		[0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
		[0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
		[0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
		[0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
		[0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
	 grad_fn=<SoftmaxBackward0>)
	'''

	context_length = attn_scores.shape[0]
	mask_simple = torch.tril(torch.ones(context_length, context_length))
	print(mask_simple)
	'''
	tensor([[1., 0., 0., 0., 0., 0.],
					[1., 1., 0., 0., 0., 0.],
					[1., 1., 1., 0., 0., 0.],
					[1., 1., 1., 1., 0., 0.],
					[1., 1., 1., 1., 1., 0.],
					[1., 1., 1., 1., 1., 1.]])
	'''

	masked_simple = attn_weights*mask_simple
	print(masked_simple)
	'''
	tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
					[0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
					[0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
					[0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
					[0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
					[0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
				 grad_fn=<MulBackward0>)
	'''

	#Normalize so that each rows ums up to one
	row_sums = masked_simple.sum(dim=1, keepdim=True)
	masked_simple_norm = masked_simple / row_sums
	print(masked_simple_norm)
	'''
	tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
					[0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
					[0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
					[0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
					[0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
					[0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
				 grad_fn=<DivBackward0>)
	'''

	#There is a smarter way to do this to avoid data leakage, cancelled out the influence of future tokens
	mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
	masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
	print(masked)
	'''
	tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
					[0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
					[0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
					[0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
					[0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
					[0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
	'''
	attn_weights=torch.softmax(masked/keys.shape[-1]**0.5, dim=-1)
	print(attn_weights)
	'''
	tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
					[0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
					[0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
					[0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
					[0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
					[0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
				 grad_fn=<SoftmaxBackward0>)
	'''

	torch.manual_seed(123)
	dropout = torch.nn.Dropout(0.5)
	example = torch.ones(6,6)
	print(dropout(example))
	'''
	tensor([[2., 2., 0., 2., 2., 0.],
					[0., 0., 0., 2., 0., 2.],
					[2., 2., 2., 2., 0., 2.],
					[0., 2., 2., 0., 0., 2.],
					[0., 2., 0., 2., 0., 2.],
					[0., 2., 2., 2., 2., 0.]])
	'''
	torch.manual_seed(123)
	print(dropout(attn_weights))
	'''
	tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
					[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
					[0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
					[0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
					[0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
					[0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]],
				 grad_fn=<MulBackward0>)
	'''


class CausalAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
		super().__init__()
		self.d_out = d_out
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))#Buffers are automatically moed to the appropriate device

	def forward(self,x):
		b, num_tokens, d_in  = x.shape
		keys = self.W_key(x)
		queries = self.W_query(x)
		values = self.W_value(x)

		attn_scores = queries @ keys.transpose(1, 2) #num_tokens and d_in dimensions
		attn_scores.masked_fill_(
			self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
		)
		attn_weights = torch.softmax(
			attn_scores / keys.shape[-1]**0.5, dim=-1
		)
		attn_weights = self.dropout(attn_weights)#Prevent overfitting

		context_vec = attn_weights @ values
		return context_vec
		

def causal_attention():
	#Ensuring the code can handle batches consisting of more than 1 input
	batch = torch.stack((inputs, inputs), dim=0)
	print(batch.shape)#torch.Size([2, 6, 3]) batchsize, token, embedding

	torch.manual_seed(123)
	context_length = batch.shape[1]
	d_in = inputs.shape[1]#B the input embedding size, d=3
	d_out = 2#C The output embedding size, d_out=2
	ca = CausalAttention(d_in, d_out, context_length, 0.0)
	context_vecs = ca(batch)
	print("context_vecs.shape:", context_vecs.shape)#context_vecs.shape: torch.Size([2, 6, 2])


#This class stacks previous CausalAttention modules
class MultiHeadAttentionWrapper(nn.Module):
	def __init__(self, d_in, d_out, context_length, dropout, num_heads,qkv_bias=False):
		super().__init__()
		self.heads = nn.ModuleList(
			[CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
			for _ in range(num_heads)]
		)
	def forward(self, x):
		return torch.cat([head(x) for head in self.heads], dim=-1)#dim=-1 as we are concatenating along the columns

def Multihead_Attention():
	torch.manual_seed(123)
	batch = torch.stack((inputs, inputs), dim=0)
	context_length = batch.shape[1] # This is the number of tokens = 6
	d_in, d_out = 3,2
	mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
	context_vecs=mha(batch)
	print(context_vecs)
	print("context_vecs.shape:", context_vecs.shape)
	'''
	tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
					 [-0.5874,  0.0058,  0.5891,  0.3257],
					 [-0.6300, -0.0632,  0.6202,  0.3860],
					 [-0.5675, -0.0843,  0.5478,  0.3589],
					 [-0.5526, -0.0981,  0.5321,  0.3428],
					 [-0.5299, -0.1081,  0.5077,  0.3493]],

					[[-0.4519,  0.2216,  0.4772,  0.1063],
					 [-0.5874,  0.0058,  0.5891,  0.3257],
					 [-0.6300, -0.0632,  0.6202,  0.3860],
					 [-0.5675, -0.0843,  0.5478,  0.3589],
					 [-0.5526, -0.0981,  0.5321,  0.3428],
					 [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
	context_vecs.shape: torch.Size([2, 6, 4])
	'''

class MultiHeadAttentionDefenitive(nn.Module):
	def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
		super().__init__()
		assert (d_out % num_heads ==0), \
		"d_out must be divisible by num_heads"
		self.d_out = d_out
		self.num_heads=num_heads
		self.head_dim=d_out // num_heads #Reduce the projection dim to match desired output dim
		self.W_query=nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key=nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value=nn.Linear(d_in, d_out, bias=qkv_bias)
		self.out_proj=nn.Linear(d_in, d_out)#Linear layer to combine head outputs
		self.dropout=nn.Dropout(dropout)
		self.register_buffer(
			"mask",
			torch.triu(torch.ones(context_length,context_length),diagonal=1)
		)

def forward(self,x):
	b, num_tokens, d_in=x.shape
	
	keys=self.W_key(x) #Shape: (b, num_tokens, d_out)
	queries=self.W_query(x)
	values=self.W_value(x)

	#Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
	keys=keys.view(b, num_tokens, self.num_heads, self.head_dim)
	values=values.view(b, num_tokens, self.num_heads, self.head_dim)
	queries=queries.view(b, num_tokens, self.num_heads, self.head_dim)

	#Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
	keys = keys.transpose(1,2)#2号和3号位交换位置
	queries = queries.transpose(1,2)
	values = values.transpose(1,2)

	# Compute scaled dot-product attention, aka self-attention with a causal mask
	attn_scores = queries @ keys.transpose(2,3) #Dot product of each head
	

	
	