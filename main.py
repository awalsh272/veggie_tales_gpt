from transformers import pipeline
from TTS.utils.synthesizer import Synthesizer

# Initialize the text generation pipeline
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

# Initialize the synthesizer with the cloned voice model
synthesizer = Synthesizer(
    tts_checkpoint="path/to/checkpoint.pth", tts_config="path/to/config.json"
)

# Sample questions
quiz = [
    {
        "question": "What is the name of the tomato in VeggieTales?",
        "answer": "Bob",
        "override_code": "VT123",
    },
    {
        "question": "What vegetable is Larry in VeggieTales?",
        "answer": "Cucumber",
        "override_code": "VT456",
    },
    {
        "question": "Who is the superhero alter ego of Larry?",
        "answer": "LarryBoy",
        "override_code": "VT789",
    },
]


def generate_question_response(question):
    prompt = f"As Bob the Tomato from VeggieTales, ask: {question}"
    response = generator(prompt, max_length=50, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]


def generate_playful_insult():
    prompt = "As Larry the Cucumber from VeggieTales, give a lighthearted remark about the incorrect answer and encourage the user to try again."
    response = generator(prompt, max_length=50, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]


def text_to_speech(text):
    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, "output.wav")


def generate_video(face_video_path, audio_path, output_path):
    import subprocess

    subprocess.run(
        [
            "python",
            "Wav2Lip/inference.py",
            "--checkpoint_path",
            "Wav2Lip/checkpoints/wav2lip_gan.pth",
            "--face",
            face_video_path,
            "--audio",
            audio_path,
            "--outfile",
            output_path,
        ]
    )


def play_video(path):
    import cv2

    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Character", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    for idx, item in enumerate(quiz):
        question = item["question"]
        correct_answer = item["answer"].lower()
        override_code = item["override_code"]
        answered_correctly = False

        while not answered_correctly:
            # Generate question response
            response = generate_question_response(question)
            print(f"Character: {response}")

            # Convert text to speech
            text_to_speech(response)

            # Generate video with lip-syncing
            generate_video(
                "path/to/character_video.mp4", "output.wav", "question_video.mp4"
            )

            # Play the video
            play_video("question_video.mp4")

            # Get user's answer
            user_input = input("You: ").strip()

            # Check for manual override
            if user_input == override_code:
                print("Manual override accepted. Moving to the next question.")
                break

            if user_input.lower() == correct_answer:
                success_response = "Correct! Let's move on to the next question."
                print(f"Character: {success_response}")

                # Convert text to speech
                text_to_speech(success_response)

                # Generate video with lip-syncing
                generate_video(
                    "path/to/character_video.mp4", "output.wav", "success_video.mp4"
                )

                # Play the video
                play_video("success_video.mp4")

                answered_correctly = True
            else:
                # Generate a playful remark
                insult_response = generate_playful_insult()
                print(f"Character: {insult_response}")

                # Convert text to speech
                text_to_speech(insult_response)

                # Generate video with lip-syncing
                generate_video(
                    "path/to/character_video.mp4", "output.wav", "insult_video.mp4"
                )

                # Play the video
                play_video("insult_video.mp4")


if __name__ == "__main__":
    main()
