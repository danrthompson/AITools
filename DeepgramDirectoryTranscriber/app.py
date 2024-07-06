import os
import click
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from dotenv import load_dotenv
from httpx import Timeout
import concurrent.futures

load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")
if API_KEY is None:
    raise ValueError("DEEPGRAM_API_KEY is not set")


def transcribe_file(input_path, output_path, deepgram):
    """Transcribe a single audio file."""
    filename = os.path.basename(input_path)
    click.echo(f"Transcribing {filename}...")

    try:
        # Open the audio file
        with open(input_path, "rb") as audio:
            audio_data = audio.read()
        payload: FileSource = {
            "buffer": audio_data,
        }
        # Set up transcription options
        options = PrerecordedOptions(
            smart_format=True,
            model="nova-2",
            dictation=True,
            language="en",
            paragraphs=True,
            punctuate=True,
            multichannel=False,
            channels=2,
        )

        # Send the audio to Deepgram for transcription
        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            payload, options, timeout=Timeout(1800, connect=10)
        )

        # click.echo(
        #     f"Done transcribing. Response: {response}. Results: {response['results']}. Metadata: {response['metadata']}"
        # )

        # Extract the transcript
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]

        trunc_transcript = "No transcript"
        try:
            trunc_transcript = transcript[0:500]
        except Exception as e:
            pass

        # click.echo(f"Transcript: {trunc_transcript}")

        # Write the transcript to a text file
        with open(output_path, "w") as output_file:
            output_file.write(transcript)

        click.echo(f"Transcript saved to {output_path}")

    except Exception as e:
        click.echo(f"Error processing {filename}: {str(e)}")


@click.command()
@click.option("--input-dir", required=True, help="Directory containing audio files")
@click.option("--output-dir", required=True, help="Directory to save transcript files")
def transcribe_files(input_dir, output_dir):
    """Transcribe audio files using Deepgram API and save as text files."""

    # Initialize Deepgram client
    deepgram = DeepgramClient(API_KEY)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of audio files
    audio_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))
    ]

    # Create a ThreadPoolExecutor with max_workers=10
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit transcription tasks
        futures = []
        for filename in audio_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, os.path.splitext(filename)[0] + ".txt"
            )
            futures.append(
                executor.submit(transcribe_file, input_path, output_path, deepgram)
            )

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    click.echo("Transcription complete!")


if __name__ == "__main__":
    transcribe_files()
