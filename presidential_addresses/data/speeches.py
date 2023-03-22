import pandas as pd

class Speeches:
	def __init__(self):
		self.speeches = self.pull_speeches()
		self.speeches_long = self.get_speeches_long()
	
	def pull_speeches(self):
		speeches = pd.read_json("https://millercenter.org/sites/default/files/corpus/presidential-speeches.json")
		speeches['title'] = speeches['title'].apply(lambda x: x.split(': ')[1])
		speeches = speeches.sort_values('date').reset_index(drop=True)
		return speeches

	def get_speeches_long(self):
		speeches_long = pd.DataFrame(columns=['title', 'date', 'president', 'text_num', 'text'])
		speeches_long = (
			self.speeches
			.assign(text=lambda x: x['transcript'].str.replace('\r','').replace('\n\n', '\n').str.split('\n'))
			.explode('text')
			.drop(columns=['transcript'])
			.query('text != ""')
			.reset_index(drop=True)
			.assign(text_num=lambda x: x.groupby('title').cumcount())
			.assign(text_len=lambda x: x['text'].str.len())
			.assign(president=lambda x: pd.Categorical(x['president'], categories=x['president'].unique()))
		)

		# for i in self.speeches.index:
		# 	speech_texts = self.speeches.loc[i, 'transcript'].replace('\r', '').replace('\n\n', '\n').split('\n')

		# 	speech_long = pd.DataFrame({
		# 		'title': self.speeches.loc[i, 'title'],
		# 		'date': self.speeches.loc[i, 'date'],
		# 		'president': self.speeches.loc[i, 'president'],
		# 		'text_num': range(len(speech_texts)),
		# 		'text': speech_texts
		# 	})
		# 	speeches_long = pd.concat([speeches_long, speech_long], ignore_index=True)

		# speeches_long['president'] = pd.Categorical(speeches_long['president'], categories=speeches_long['president'].unique())
		# speeches_long['text_len'] = speeches_long['text'].apply(len)
		return speeches_long

if __name__ == "__main__":
	speech_data = Speeches()
	speech_data.speeches.loc[:, ['title', 'transcript']]
	print(speech_data.speeches_long.loc[:10, ['title', 'text', 'text_num', 'text_len']])