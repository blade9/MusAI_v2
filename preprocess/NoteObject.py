
class NoteObject:

    def __init__(self, duration, start_time, time_num=4, time_denom=4):
        self.duration = duration
        self.note_type = self.calcClosestNoteType(duration/time_denom)
        self.starting_time = start_time


    NOTE_TYPES = {
        "whole": 1,
        "half": 0.5,
        "quarter": 0.25,
        "eighth": 0.125,
        "sixteenth": 0.0625,
        "thirty-second": 0.03125
    }

    def getNoteType(self):
        return self.note_type

    def calcNoteType(self, duration_in_beats):
        for note_type, threshold in NoteObject.NOTE_TYPES.items():
            if duration_in_beats >= threshold:
                return note_type
        return "none"

    def calcClosestNoteType(self, duration_in_beats):
        closest_note = None
        min_diff = float('inf')

        for note_type, note_duration in NoteObject.NOTE_TYPES.items():
            diff = abs(duration_in_beats - note_duration)

            if diff < min_diff:
                closest_note = note_type
                min_diff = diff

        return closest_note


    def getStartTime(self):
        return self.starting_time

    def getDuration(self):
        return self.duration

    def getData(self):
        return self.duration, self.note_type, self.starting_time




