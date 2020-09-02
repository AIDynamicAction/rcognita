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

	def save_to_json(self, assignment_name, first_name, last_name):
		file_n = f"{last_name}_{first_name}_{assignment_name}"

		with open(file_n+'_answers.json', 'w') as answers_file:
			json.dump(self.answers_dict, answers_file)

	# @classmethod
	# def grade_assignment(cls, answer_key, student_version):
	# 	files = [answer_key, student_version]
	# 	decoded_files = []
		
	# 	for file in files:
	# 		with open(file, 'r') as json_file:
	# 			decoded = json.load(json_file)
	# 			decoded_files.append(decoded)

	# 	for answers_dict, student_dict in decoded_files:
			
			