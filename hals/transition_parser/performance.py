def measure_performance(sentences, parses):
    correct_labels = 0
    correct_edges = 0
    total = 0

    for sent, parsed_sent in zip(sentences, parses):
        heads, labels = parsed_sent
        total += len(heads)
        num_unlabeled, num_labeled = sent.num_correct(heads, labels)
        correct_edges += num_unlabeled
        correct_labels += num_labeled

    return {
        'ua': correct_edges,
        'la': correct_labels,
        'total': total,
        'uas': correct_edges / total,
        'las': correct_labels / total
    }

