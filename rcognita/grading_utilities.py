import json

class AnswerTracker:
	def __init__(self):
		self.answers_dict = {}

	def record(self, problem, answer):
		self.answers_dict[problem] = answer

	def delete(self, problem):
		try:
			del self.answers_dict[problem]
		
		except KeyError:
			print("Key not found")

	def print_answers(self):
		print(self.answers_dict)

	def save_to_json(self):
		with open('answers.json', 'w') as answers_file:
			json.dump(self.answers_dict, answers_file)