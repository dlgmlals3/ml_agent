from mlagents_envs.environment import UnityEnvironment

if __name__ == '__main__':
    # 환경을 정의    
    env = UnityEnvironment(file_name='C:\work\ml_agent\mlProject\\build')
     
    # behavior 불러오기
    env.reset() # 환경 초기화
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f'name of behavior :  {behavior_name}')
    # 에이전트의 행동이나 관측의 대한 정보 포함
    spec = env.behavior_specs[behavior_name]

    # 에피소드 진행을 위한 반복문 (10에피소드 반복)
    for ep in range(10):
        print(f'1 name of behavior :  {behavior_name}')
        env.reset() # 환경초기화
        # 에이전트가 행동을 요청한 상태인지, 마지막 상태인지 확인
        # 스텝에서 에이전트의 정보 (보상, 행동, 상태)를 반환
        # 에피소드의 마지막 스텝일 경우 terminal_step 에 저장
        # 다음 행동을 요청한 스텝의 대한 정보는 decision_step 에 저장
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        print(f'2 name of behavior :  {behavior_name}')
        # 한 에이전트를 기준으로 로그를 출력
        tracked_agent = -1 # 추적할 에이전트 ID
        done = False # 한 에피소드 마무리 되었는지 판단
        ep_rewards = 0 # 한 에피소드의 보상의 총합


        while not done:
            # tracked agent 지정
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]
            # 랜덤 액션 결정  
            action = spec.action_spec.random_action(len(decision_steps))
            # set actions 같은 behavior_name 가진 에이전트의 그룹의 행동을 정의
            env.set_actions(behavior_name, action)
            # 실제 액션 수행
            env.step()

            #텝 종류 후 에이전트의 정보 (보상, 상태 ) 취득
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            # 추적중인 에이전트가 행동이 가능한 상태와 종료 상태일때를 구분하여 보상 저장
            if tracked_agent in decision_steps:
                ep_rewards += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps:
                ep_rewards += terminal_steps[tracked_agent].reward
                done = True
            # 한 에피소드가 종료되고 추적중인 에이전트의 대해서 해당 에피소드에서의 보상 출력
        print(f'total reward for ep {ep} is {ep_rewards}')
env.close()