from Models.lstmModel.NoteObject import NoteObject

class BeatObject:
    def __init__(self, numberID, notes, tempo, beats_per_measure=4, note_value_per_beat=4):
        self.numberID = numberID
        self.notes = notes
        self.tempo = tempo
        self.beats_per_measure = beats_per_measure
        self.note_value_per_beat = note_value_per_beat

    def addNote(self, note):
        self.notes.append(note)
        self.notes.sort(key=lambda n: n.starting_time)

    def getID(self):
        return self.numberID

    def getLSTMoutputForm(self):
        for my_note in self.notes:

            pass


