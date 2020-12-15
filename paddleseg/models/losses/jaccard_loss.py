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

from paddleseg.cvlibs import manager
from .iou import compute_predictions_iou


@manager.LOSSES.add_component
class JaccardLoss(nn.Layer):
    """
    Implements the tversky loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=255):
        super(JaccardLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5

    def forward(self, logits, labels):
        intersection, cardinality = compute_predictions_iou(logits, labels, self.ignore_index)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + self.eps)).mean()
        return 1 - jacc_loss
