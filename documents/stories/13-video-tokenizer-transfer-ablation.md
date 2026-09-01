As an engineer testing whether natural-video pretraining transfers to numerical physics fields, I want a controlled tokenizer ablation, so that we learn whether the transferable representation is useful before downloading or fine-tuning a multi-billion-parameter dynamics model.

The first milestone MUST be a model-independent data and evaluation harness: a versioned dataset slice, fixed train/validation/test split, field names, units, normalization transforms, boundary conditions, compression ratio, metrics, and trivial reconstruction baselines. It MUST run before any multi-billion-parameter checkpoint is downloaded or any cloud training is rented.

The ablation MUST compare a video-pretrained tokenizer with the identical tokenizer architecture initialized from scratch, using the same data, channel mapping, optimizer, training-step budget, seeds, and evaluation. Initialization MUST be the only intended experimental difference.

Adapting RGB input and output layers to physics-field channels MUST preserve every compatible pretrained weight and MUST record exactly which parameters were retained, inflated, replaced, frozen, or trained. A run that discards most of the pretrained representation MUST NOT be reported as evidence about transfer.

Evaluation MUST report reconstruction error per physical field, error in physical units after denormalization, compression rate, training compute, inference latency, and field-appropriate structure such as spectra or conserved integral quantities. It MUST include multiple seeds and MUST NOT collapse all fields into one favorable average.

A go/no-go margin MUST be fixed before training. The larger autoregressive experiment MUST NOT begin unless the pretrained tokenizer clears that margin on held-out data without losing physical structure, and a negative or inconclusive result MUST be retained with the same artifacts as a positive result.

Never choose the dataset, metric, or stopping point after seeing which initialization it favors.

For example, begin with a small, fixed slice of The Well's `turbulent_radiative_layer_2D` data and prove the loader, inverse normalization, metrics, and identity or low-capacity autoencoder baseline locally. Only then compare the channel-adapted Cosmos video tokenizer against the same tokenizer trained from scratch on the RTX 3060; do not download the 4B autoregressive checkpoint for this story.

