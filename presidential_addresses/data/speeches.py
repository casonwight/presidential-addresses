import pandas as pd

def get_speeches():
	speeches = pd.read_json("https://millercenter.org/sites/default/files/corpus/presidential-speeches.json")
	speeches['title'] = speeches['title'].apply(lambda x: x.split(': ')[1])
	speeches = speeches.sort_values('date')
	return speeches

if __name__ == "__main__":
	speeches = get_speeches()
	print(speeches)