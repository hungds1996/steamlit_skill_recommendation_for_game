import streamlit as st
import pandas as pd
from typing import Dict

from utils import *
from skill_recommendation import get_model
from constants import passive_skill, all_skill, evolve_requirement


def main():
    st.title("Skill recommendation")
    
    if 'selected_skills' not in st.session_state:
        st.session_state['selected_skills'] = ['Revolver']
    if 'pred_states' not in st.session_state:
        st.session_state['pred_states'] = None
    if 'predict_next_skill' not in st.session_state:
        st.session_state['predict_next_skill'] = False
    if 'skip_skills' not in st.session_state:
        st.session_state['skip_skills'] = ["end"]
    if 'prediction_logits' not in st.session_state:
        _, _, predicted_logits = one_step_model.generate_one_step(st.session_state['selected_skills'], states=None, temperature=0.1)
        
        predicted_logits = predicted_logits.numpy().tolist()[0]
        
        df = pd.DataFrame(list(zip(all_skill, predicted_logits)), columns=['Skill', "Score"])
        st.session_state['prediction_logits'] = df
    temperature = st.sidebar.slider("Select a degree of random ", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    options = all_skill
    selected_option = st.selectbox("select a skill", options)

    col1, col2, _, _, _ = st.columns(5)
    
    if col1.button("Add to List", key="add_button", type='primary'):
        print(selected_option)
        st.session_state['selected_skills'].append(selected_option)
        st.session_state['predict_next_skill'] = True
    
    if col2.button("Delete Last Item", key="delete_button", help="Click to delete last item", type='secondary'):
        st.session_state['selected_skills'].pop()
        st.session_state['predict_next_skill'] = True
    
    states = st.session_state['pred_states']
    
    if len(st.session_state['selected_skills']) >= 1 and st.session_state['predict_next_skill']:
        input = st.session_state['selected_skills']
        
        skills_to_skip = get_skills_to_skip(st.session_state['selected_skills'])
        one_step_model.set_chars_to_skip(skills_to_skip)
        
        _, states, predicted_logits = one_step_model.generate_one_step(input, states=states, temperature=temperature)
        
        st.session_state['pred_states'] = states
        predicted_logits = predicted_logits.numpy().tolist()[0]
        
        df = pd.DataFrame(list(zip(all_skill, predicted_logits)), columns=['Skill', "Score"])
        st.session_state['prediction_logits'] = df


    st.write("Current skill list: ")
    st.write(transform_skill_list(st.session_state['selected_skills']))
    st.sidebar.write(st.session_state['prediction_logits'].sort_values(by="Score", ascending=False).reset_index(drop=True))
    st.session_state['predict_next_skill'] = False
    
    
def on_button_click(state, item):
    state.selected_items.append(item)
    
if __name__ == "__main__":
    active_skill = [s for s in all_skill if s not in passive_skill and s != 'end']
    one_step_model = get_model()
    
    main()