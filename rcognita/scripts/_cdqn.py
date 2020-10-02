from rcognita import ENDI2, ExperienceBuffer, Agent
import argparse

def main(args=None):
    REPLAY_SIZE = 10000
    sys = ENDI2()
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(sys, buffer)
    agent.train_sim()

if __name__ == "__main__":
	main()
