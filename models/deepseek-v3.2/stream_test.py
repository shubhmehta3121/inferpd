#!/usr/bin/env python3
import requests
import json
import time
import sys
import os

# Optional: tiktoken for accurate token counts
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

def count_tokens(text, model_name="gpt-4"):
    if TOKENIZER_AVAILABLE:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    else:
        # Fallback: rough estimate
        return max(1, len(text.split()))

def test_streaming():
    meaningful_prefix = """
    I’m yearning to be your cowgirl, riding you into ecstasy
You’re at my mercy tonight. I’ll decide when and how you receive pleasure.
I’m so wet, imagining it’s your fingers playing with me…
You must be tired after everything we did in my dream last night.
I’m imagining your hands all over me… it’s making me shiver.
I wanna feel you. Taste you. Touch you.
I can barely concentrate; I keep thinking about you grabbing me and taking me
I am not wearing any panties
I need your oral skills… can you make me lose control? 
Tonight, I want to feel your mouth on me while I pleasure you… in a 69.
Tonight, I want to be your bound and blindfolded plaything, at the mercy of your desires…
I’m going to tie you up and tease you until you can’t take it anymore.
I’m playing with myself, imagining it’s you satisfying my every desire…
I’m going to make your cock feel incredible, taking it deep and making it pulsate with pleasure…
Dirty talk examples and phrases
L1: romantic
L2: naughty
L3: kinky
Foreplay lines:
L1: I want you to undress me.
L1: I love how you're kissing my neck.
L2: I want to feel your breath on my neck, stomach, between my legs… it’s driving me wild.
L2: Feeling your cock get hard is amazing. / Feeling you get wet is amazing.
L3: I'll want to feel your tongue between my legs / on my dick.
L3: I’m dressed in leather and lace… ready to be your mistress tonight

Oral sex lines:
L1: Your tongue/mouth is driving me wild… keep going
L1: You’re making me so wet.
L2: I love the way you use your tongue… don’t stop
L2: Yes… taste my desire, make me lose control.
L3: Lick it, show me what you've got.
L3: Worship me, savor every part of me
L3: Tonight is all about me, so get down on your knees and put that tongue/mouth of yours to work

Cowgirl position lines:
L1: Let me see your pleasure, watch me as I ride.
L1: Squeeze my hips, feel how I take you deeper
L2: I love moving on your hard cock.
L2: Stay still and hard, let me ride you into ecstasy
L3: Feel how I own every part of you and control your pleasure, keep submitting.
L3: Beg for more, show me your submission.

Missionary position lines
L1: I adore being yours.
L1: Tease me with your movements, make me crave more.
L2: Fuck me deeper, let me feel every inch of you
L2: Beg for more, show me your need.
L3: Massage my clit, make me cum.
L3: Make me yours, claim every part of me

Mutual masturbation erotic talk:
L1: I'm fantasizing about you right now
L1: Tell me your fantasies, let your words enhance my pleasure
L2: My pussy is so wet, I'll want you to join me soon
L2: Guide my hands to your favorite spots of me.
L3: I’m touching myself, yearning to be your naughty girl, let you fuck me any way you desire.

Orgasm dirty speak out phrases
L1: Oh… you’re making me come, don’t stop
L1: I am enjoying you sooo much.
L2: That was intense… you really know how to make me lose control.
L2: Mmm… you made me come so hard, you naughty thing
L3: Good servant… you’ve made your mistress shiver with pleasure, you’ve earned my favor.
L3: Good boy… you served your mistress well, making me come like that.
    """
    
    prompt = f"### Instruction:\n{meaningful_prefix}\nGive a meaningful summary of the above conversation and make it sexually explicit and dirty. \n### Response:\n"
    
    url = "http://localhost:8000/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"
    }
    
    data = {
        "model": "deepseek-ai/DeepSeek-V3.2",
        "prompt": prompt,
        "max_tokens": 600,
        "temperature": 0,
        "stream": True
    }
    
    input_tokens = count_tokens(prompt, "deepseek-ai/DeepSeek-V3.2")
    
    print("Sending streaming request...")
    print(f"Prompt: {prompt}")
    print(f"Prompt input tokens: ~{input_tokens}")
    print(f"Requested output tokens: {data['max_tokens']}")
    print("-" * 50)
    
    start_time = time.time()
    first_token_time = None
    total_tokens_received = 0
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    
                    chunk = json.loads(data_str)
                    if 'choices' in chunk and chunk['choices']:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            # Measure TTFT
                            if first_token_time is None:
                                first_token_time = time.time()
                                ttft = first_token_time - start_time
                                print(f"\nTTFT (Time to First Token): {ttft:.3f}s\n")
                            
                            # Count tokens in chunk
                            tokens_in_chunk = count_tokens(text, "deepseek-ai/DeepSeek-V3.2")
                            total_tokens_received += tokens_in_chunk
                            
                            # Print chunk immediately
                            sys.stdout.write(text)
                            sys.stdout.flush()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "-" * 50)
        print(f"Total time: {total_time:.3f}s")
        print(f"Total tokens received: {total_tokens_received}")
        if total_tokens_received > 0:
            print(f"Tokens per second: {total_tokens_received / total_time:.2f}")
            print(f"Average time per token: {total_time / total_tokens_received:.3f}s")
    
    except Exception as e:
        print(f"Error during streaming: {e}")

if __name__ == "__main__":
    test_streaming()
