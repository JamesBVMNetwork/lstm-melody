import argparse
import pandas as pd
import json
from music21 import note, stream, chord

def chords_n_notes(Snippet):
    Melody = []
    offset = 0 
    for i in Snippet:
        print(i.isdigit())
        #If it is chord
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".") #Seperating the notes in chord
            notes = [] 
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)            
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
        # pattern is a note
        else: 
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    Melody_midi = stream.Stream(Melody)   
    return Melody_midi
    
def create_midi(prediction_output, output_file='test_output.mid'):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    
    Melody_midi = chords_n_notes(prediction_output)
    Melody_midi.write('midi', fp=output_file)

parser = argparse.ArgumentParser(description='Export midi from note sequence')
parser.add_argument('--data', type=str, default='[]', help='Note sequences')
parser.add_argument('--output-path', type=str, default='output.mid', help='Path to export midi file')
args = parser.parse_args()

note_sequence = json.loads(args.data)
print(note_sequence[0])
output_path = args.output_path
create_midi(note_sequence, output_path)