# Upstream source and model assets

The adapter in this directory is distributed under [MIT](LICENSE), as part of
Mosaic. It invokes external DeFoG source in a separate interpreter; no upstream
source tree, checkpoint, dataset, or generated molecular archive is bundled.

DeFoG is by Yiming Qin, Manuel Madeira, Dorina Thanou, and Pascal Frossard:
[official repository](https://github.com/manuelmlmadeira/DeFoG),
[paper](https://arxiv.org/abs/2410.04263), ICML 2025. The default adapter targets
source revision `365bda9affadd5c2307014a0532ddaa244399441`. The official repository
links its [checkpoint collection](https://drive.switch.ch/index.php/s/MG7y2EZoithAywE).
Obtain model assets from that source and retain any asset-specific notices;
this package's MIT license does not grant additional rights in external assets.

The following is the upstream source's MIT notice at that revision, available
in its [LICENSE](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/LICENSE):

```text
Copyright (c) 2012-2025 Manuel Madeira, Yiming Qin

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
