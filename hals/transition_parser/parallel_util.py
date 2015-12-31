from collections import namedtuple
from queue import Queue
from threading import Thread

from tensorflow.python.training.coordinator import Coordinator


class ParallelParse:
    """
    Data structures for parsing in parallel via threads
    """
    def __init__(self, sentences, score_batch, advance_batch, prepare_feed, transition_system, batch_size=64):
        self.sentences = sentences
        self.batch_size = batch_size
        self.batches = Queue()
        self.needs_scoring = Queue()
        self.num_left = len(sentences)
        self.states = []
        self.transition_system = transition_system

        self.prepare_feed = prepare_feed
        self.score_batch = score_batch
        self.advance_batch = advance_batch

    def parse(self):
        t_sys = self.transition_system
        # Put all sentences in their initial states and queue them for scoring
        for sent_i, sent in enumerate(self.sentences):
            self.states.append(t_sys.state(len(sent)))
            self.needs_scoring.put(sent_i)

        self._run_threads()

        return [t_sys.extract_parse(state) for state in self.states]

    def _run_threads(self):
        # Setup threads
        coord = Coordinator()

        t_prepare_batch = Thread(target=prepare_batches, args=[coord, self])
        t_score_batch = Thread(target=score_batches, args=[coord, self])

        threads = [t_prepare_batch, t_score_batch]

        for t in threads:
            t.start()
        coord.join(threads)

    def allowed_batch(self, states):
        allowed_list = []
        for state in states:
            allowed_list.append(self.transition_system.allowed(state))
        return allowed_list

SentenceBatch = namedtuple('SentenceBatch', 'ids feed_dict sents states allowed_list')


def prepare_batches(coord: Coordinator, pp: ParallelParse):
    def build_batch(sent_ids):
        sents = [pp.sentences[sent_id] for sent_id in sent_ids]
        states = [pp.states[sent_id] for sent_id in sent_ids]

        return SentenceBatch(ids=sent_ids,
                             sents=sents,
                             states=states,
                             feed_dict=pp.prepare_feed(sents, states),
                             allowed_list=pp.allowed_batch(states)
                             )
    sent_ids = []
    while True:
        item = pp.needs_scoring.get()
        if item is None:
            if coord.should_stop():
                break
            else:
                if len(sent_ids) >= pp.num_left:
                    pp.batches.put(build_batch(sent_ids))
                    sent_ids = []
        else:
            sent_ids.append(item)
            if len(sent_ids) >= min(pp.batch_size, pp.num_left):
                pp.batches.put(build_batch(sent_ids))
                sent_ids = []


def score_batches(coord: Coordinator, pp: ParallelParse):
    t_sys = pp.transition_system
    while True:
        batch = pp.batches.get()

        # Score states and move to next states, based on the scoring
        batch_scores = pp.score_batch(batch.feed_dict, batch.sents, batch.states, batch.allowed_list)
        pp.advance_batch(batch.ids, pp.states, batch_scores, batch.allowed_list)

        # Put non-finished sentences back in scoring queue
        queue_again = [sent_id for sent_id in batch.ids
                       if not t_sys.is_final(pp.states[sent_id])]
        pp.num_left -= (len(batch.ids) - len(queue_again))
        for sent_id in queue_again:
            pp.needs_scoring.put(sent_id)

        # Wake the batch prepare thread up and give it a chance to enqueue
        # a smaller batch if need be
        if len(queue_again) == 0 and pp.num_left > 0:
            pp.needs_scoring.put(None)

        # Our work here is done. Stop the other thread.
        if pp.num_left == 0:
            coord.request_stop()
            pp.needs_scoring.put(None)
            break
