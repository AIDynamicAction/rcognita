import json


class AnswerTracker:
    def __init__(self):
        self.answer_dict = {}

    def record(self, problem, answer):
        self.answer_dict[problem] = answer

    def delete(self, problem):
        try:
            del self.answer_dict[problem]

        except KeyError:
            print("Key not found")

    def print_answers(self):
        print(self.answer_dict)

    def save_to_json(self, assignment_name, first_name, last_name):
        file_n = f"{last_name}_{first_name}_{assignment_name}"

        with open(file_n + '.json', 'w') as submitted_answers:
            json.dump(self.answer_dict, submitted_answers)


def load_from_json(filename):
    f = open(filename, )

    data = json.load(f)
    f.close()

    return data
