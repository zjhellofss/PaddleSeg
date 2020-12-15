# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn
import paddle.nn.functional as F


def compute_predictions_iou(logits, labels, ignore_index):
    if len(labels.shape) != len(logits.shape):
        labels = paddle.unsqueeze(labels, 1)
    num_classes = logits.shape[1]
    mask = (labels != ignore_index)
    logits = logits * mask
    labels = paddle.cast(labels, dtype='int32')
    single_label_lists = []
    for c in range(num_classes):
        single_label = paddle.cast((labels == c), dtype='int32')
        single_label = paddle.squeeze(single_label, axis=1)
        single_label_lists.append(single_label)
    labels_one_hot = paddle.stack(tuple(single_label_lists), axis=1)
    logits = F.softmax(logits, axis=1)
    labels_one_hot = paddle.cast(labels_one_hot, dtype='float32')
    dims = (0,) + tuple(range(2, labels.ndimension()))
    intersection = paddle.sum(logits * labels_one_hot, dims)
    cardinality = paddle.sum(logits + labels_one_hot, dims)
    mask.stop_gradient = True
    labels.stop_gradient = True
    return intersection, cardinality
