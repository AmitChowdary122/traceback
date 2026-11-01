# data/sample: provenance and license

## What this is

This is a cut-down excerpt of the **ALEAPP report** for a public Android 14 reference image, plus a small set of media
files from the same extraction. Every row in `aleapp/_TSV Exports/` is real ALEAPP output, copied verbatim. No row has
been added or edited, and no file has been synthesized.

| | |
|---|---|
| Image | **Android 14 Public Image**, created by **Joshua Hickman** (thebinaryhick.blog) |
| Distributed by | Digital Corpora (sponsored by the AWS Open Data Sponsorship Program) |
| Download | `https://digitalcorpora.s3.amazonaws.com/corpora/mobile/android_14/Android_14_Public_Image.tar.gz` |
| Archive size | 18,411,199,735 bytes. S3 `Last-Modified` Sat, 14 Sep 2024 18:55:54 GMT; S3 ETag `f70914841a4bc9ff050f6f17dae4c7ce-2195` (multipart, not a content hash) |
| Archive SHA-256 | `4e1698460550e20f3b334ae9df96ee1cd78314bdc0fe5de5183fd5d7e1c8967e` (computed by streaming the whole archive on Nov 1, 2026; 18,411,199,735 bytes read, equal to the S3 Content-Length. Digital Corpora publishes no hash for this image) |
| Documentation | `https://digitalcorpora.s3.amazonaws.com/corpora/mobile/android_14/Android14-ImageCreation.pdf` (SHA-256 `49dc2cb39f8900da9b7167455183a01cbf07e3db8b211798fd1922ab01a2283a`) |
| Device | Google Pixel 7a (`lynx`), Android 14, build UQ1A.240105.004, Google Fi |
| Extraction | Cellebrite UFED full file system, `UFED Google Pixel 7a 2024_07_28 (001)/EXTRACTION_FFS 01/EXTRACTION_FFS.zip` (34,341,538,452 bytes) inside the archive |
| Persona | "Liz Dehner" (`ldehner505@gmail.com`), populated by the image creator between 2024-01-26 and 2024-07-28 |
| Case timezone | `America/New_York`, from `persist.sys.timezone` in `data/property/persistent_properties`. This matches the documentation: EST until 2024-03-10, EDT after |

## How it was made

1. **Streamed, not stored.** The 18 GB archive did not fit next to its 34 GB inner zip on the build machine. It was
   piped through `gunzip` → `tar -xO` → libarchive's streaming zip reader. Only these extraction paths were written to
   disk: `data/data`, `data/user_de/0`, `data/system*`, `data/misc*`, `data/media/0`, `data/property` and `*build.prop`.
   APK, ODEX/VDEX/OAT, `.so`, `code_cache`, `app_webview` and thumbnails were skipped, along with any single file over
   300 MB (2 GB for DCIM/Pictures). 124,949 zip members were read; 87,263 files were kept (9.9 GB).
2. **ALEAPP** `2026.4.5-dev`, at commit `0855bbd0301ab86e8a287b166263da4d96f881f2` of `github.com/abrignoni/ALEAPP`,
   on Python 3.12: `python aleapp.py -t fs -i <extracted fs> -o <out>`. All 1,539 modules ran, and the report took
   about 5 minutes. ALEAPP renders all times in UTC. Spot checks against the creation document agree: the Google
   Messages "Are you here yet?" is `2024-07-26 19:03:11Z` (documented as 15:03 EDT), and the outgoing call to
   919-579-4674 is `2024-07-27 21:40:04Z`, 121 s (documented as 17:40 EDT, 2:01).
3. **Cut-down.** `scripts/build_sample.py` writes this directory. `MANIFEST.json` lists the SHA-256 of every file here
   and, for media, the SHA-256 and size of the original file in the extraction.

## Contents

`aleapp/`: the report root. Its `_TSV Exports/` files are copied byte for byte from the full report, so TSV row
numbers, and the artifact IDs derived from them, are the same as in the full report.

| Family | TSVs (rows) |
|---|---|
| messages | `GoogleMessages.tsv` (1,123), `WhatsApp - One To One Messages.tsv` (73), `WhatsApp - Group Messages.tsv` (156), `Telegram - Messages.tsv` (34), `Facebook Messenger - Chats (msys_database).tsv` (26), `Google Voice - Messages.tsv` (14) |
| calls | `Call logs .tsv` (2,655; most are Google Fi voicemail stubs), `WhatsApp - Call Logs.tsv` (4), `Facebook Messenger - Calls (msys_database).tsv` (8), `Google Voice - Calls.tsv` (4) |
| locations | `ODLH Path Points.tsv` (14,207; on-device Google Timeline), `Life360 - Locations.tsv` (15,668), `Fitbit - Exercise GPS.tsv` (3,366) |
| support | `Contacts.tsv` (names for call numbers), `Build.tsv` (device model), `_lava_data.lava` (ALEAPP run metadata and version) |

`media/`: an extraction-root mirror, so each file sits at its path in the image.

| Path | Files | Notes |
|---|---|---|
| `data/media/0/DCIM/Camera/PXL_*.jpg` | 40 | Every camera photo dated 2024-06-15 or later; 37 are geotagged. **Downscaled** to at most 1600 px (JPEG q85), with EXIF kept, including DateTimeOriginal, OffsetTimeOriginal and GPS. The originals' hashes are in `MANIFEST.json`. |
| `data/data/com.google.android.apps.messaging/cache/telephony_cache/*.bin` | 26 | Google Messages image, GIF and video attachments, **verbatim**. They include the two MMS videos (2024-02-09 and 2024-02-13) whose audio tracks are the only recorded audio in the image. |
| `data/property/persistent_properties` | 1 | Verbatim. It is the source of `case_meta.device_tz`. |

Left out on purpose: a 74-page PDF received as a message attachment (third-party copyrighted document, 7 MB). The
rest of the extraction and the full report are also left out. The full report holds Wi-Fi pre-shared keys and other
credentials of the test persona, and none of that is in this sample.

## Known gaps

- **No voice notes.** Neither the Android 14 documentation nor the file listing of the extraction has any voice
  message, voice memo or recorder file; the same is true of the Android 10 and 12 documentation. The 2,154 Google Fi
  voicemail files in `com.android.providers.contacts/app_voicemail-data/` are all **0 bytes**: the audio stays on
  the carrier's server. Only Google's transcription text for some of them survives, in the `Transcription` column of
  `Call logs .tsv`. The only recorded audio in this sample is the soundtrack of the two MMS video attachments.
- Signal's database is encrypted, so ALEAPP produced no Signal rows.
- WhatsApp group history includes old messages (from 2017 on) that synced in when the persona joined spam groups.

## License and terms

- **Digital Corpora terms of use** (digitalcorpora.org/about-digitalcorpora/terms-of-use, read Nov 1, 2026):
  "Unless otherwise notice, all original content on this site, including metadata, original scenario data, and
  teachers guides, are made available under the CC0 license … Any material that explicitly claims a copyright is not
  subject to the CC0 license. This includes copyrighted software and other original materials located in disk images,
  packet dumps, and other digital artifacts located on this website."
- The image itself, its persona data and its documentation were created by Joshua Hickman and published for research
  and training through Digital Corpora.
- Third-party content inside the image is not covered by CC0. That includes app assets, marketing images received by
  SMS, and posts. The few such items here are message attachments kept as evidence excerpts for research and
  demonstration, and they belong to their owners.
- ALEAPP is MIT-licensed (github.com/abrignoni/ALEAPP). Its TSV output is reproduced here.
- Traceback's own code is Apache-2.0 (see `/LICENSE`). That license does not apply to the files in `data/sample/`.
