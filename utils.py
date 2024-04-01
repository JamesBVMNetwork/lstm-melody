from music21 import converter, note, chord, instrument


def read_notes_from_midi(file):
    notes = []
    midi = converter.parse(file)
    songs = instrument.partitionByInstrument(midi)
    for part in songs.parts:
        pick = part.recurse()
        for element in pick:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))
    return notes