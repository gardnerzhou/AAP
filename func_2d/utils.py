""" helper function

author junde
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import cfg


args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)



def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if net == 'sam2':
        from sam2_train.build_sam import build_sam2
        # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        # if torch.cuda.get_device_properties(0).major >= 8:
        #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
        # æ ¹æ®å‚æ•°é€‰æ‹©æ˜¯å¦ä½¿ç”¨ä¿®æ”¹åçš„mask decoder
        use_modified_mask_decoder = getattr(args, 'use_modified_mask_decoder', True)
        net = build_sam2(args.sam_config, args.sam_ckpt, device="cuda", use_modified_mask_decoder=use_modified_mask_decoder)


    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net


def create_logger(log_dir, phase='train'):
    # time_str = time.strftime('%Y-%m-%d-%H-%M')
    # log_file = '{}_{}.log'.format(time_str, phase)
    log_file = 'train.log'
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger



def eval_seg(pred, true_mask_p, threshold):
    """
    è®¡ç®—åˆ†å‰²è¯„ä¼°æŒ‡æ ‡
    Args:
        pred: é¢„æµ‹ç»“æœ
        true_mask_p: çœŸå®æ ‡ç­¾
        threshold: é˜ˆå€¼
    Returns:
        Iou: äº¤å¹¶æ¯”
        Dice: Diceç³»æ•°
    """
    smooth = 1e-8
    
    # è½¬æ¢ä¸ºnumpy
    pred = pred.cpu().numpy()
    true_mask_p = true_mask_p.cpu().numpy()
    
    # è®¡ç®—æ¯ä¸ªæ ·æœ¬çš„æŒ‡æ ‡
    total_iou = 0
    total_dice = 0
    batch_size = pred.shape[0]
    
    for i in range(batch_size):
        # è·å–å½“å‰æ ·æœ¬
        curr_pred = pred[i].flatten()
        curr_true = true_mask_p[i].flatten()
        
        # è®¡ç®—äº¤é›†
        intersection = (curr_pred * curr_true).sum()
        
        # è®¡ç®—IoU
        iou = (intersection + smooth) / (curr_pred.sum() + curr_true.sum() - intersection + smooth)
        
        # è®¡ç®—Dice
        dice = (2 * intersection + smooth) / (curr_pred.sum() + curr_true.sum() + smooth)
        
        total_iou += iou
        total_dice += dice
    
    # è¿”å›å¹³å‡å€¼
    return total_iou / batch_size, total_dice / batch_size




class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        import logging
        logger = logging.getLogger('train_debug')
        
        # ç›‘æ§è¾“å…¥
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logger.error(f"[DiceLoss debug] inputs contains NaN/Inf before sigmoid")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logger.error(f"[DiceLoss debug] targets contains NaN/Inf")
        
        # å°†è¾“å…¥é€šè¿‡sigmoidå‡½æ•°ï¼ˆå¦‚æœæ˜¯äºŒåˆ†ç±»é—®é¢˜ï¼‰
        inputs = torch.sigmoid(inputs)
        
        # ç›‘æ§sigmoidè¾“å‡º
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logger.error(f"[DiceLoss debug] inputs contains NaN/Inf after sigmoid")
        
        # å±•å¹³è¾“å…¥å’Œç›®æ ‡å¼ é‡
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # è®¡ç®—äº¤é›†å’Œå¹¶é›†
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        # ç›‘æ§äº¤é›†å’Œå¹¶é›†
        if torch.isnan(intersection).any() or torch.isinf(intersection).any():
            logger.error(f"[DiceLoss debug] intersection contains NaN/Inf")
        if torch.isnan(union).any() or torch.isinf(union).any():
            logger.error(f"[DiceLoss debug] union contains NaN/Inf")
        
        # è®¡ç®—Diceç³»æ•°
        dice_numerator = 2. * intersection + self.smooth
        dice_denominator = union + self.smooth
        
        # ç›‘æ§åˆ†æ¯
        if dice_denominator <= 0:
            logger.error(f"[DiceLoss debug] dice_denominator <= 0: {dice_denominator.item()}")
        
        dice = dice_numerator / dice_denominator
        
        # ç›‘æ§Diceç³»æ•°
        if torch.isnan(dice).any() or torch.isinf(dice).any():
            logger.error(f"[DiceLoss debug] dice coefficient contains NaN/Inf")
            logger.error(f"[DiceLoss debug] intersection: {intersection.item()}, union: {union.item()}")
        
        # è¿”å›DiceæŸå¤±
        dice_loss = 1 - dice
        
        # ç›‘æ§æœ€ç»ˆæŸå¤±
        if torch.isnan(dice_loss).any() or torch.isinf(dice_loss).any():
            logger.error(f"[DiceLoss debug] dice_loss contains NaN/Inf")
        
        return dice_loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation (TV) Regularization Loss.
    Encourages smoothness in the segmentation output by penalizing large gradients.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super(TotalVariationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        """
        Compute the total variation loss for a batch of segmentation outputs.

        Args:
            x (torch.Tensor): Segmentation output tensor of shape (B, C, H, W),
                              where B is the batch size, C is the number of channels,
                              and H, W are the height and width of the image.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute horizontal gradient (difference along the width dimension)
        delta_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        # Compute vertical gradient (difference along the height dimension)
        delta_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        # Combine the two gradients
        tv_loss = torch.mean(delta_x) + torch.mean(delta_y)

        # Optionally apply reduction
        if self.reduction == 'sum':
            tv_loss = tv_loss * x.size(0)  # Multiply by batch size to sum over the batch

        return tv_loss

class BoundaryTotalVariationLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BoundaryTotalVariationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        # ä½¿ç”¨ Sobel ç®—å­æå–è¾¹ç•Œ
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        if x.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()

        # è®¡ç®—æ´´å¹³å’Œåç›´æ–¹å‘çš„æ¢‡åº¦
        grad_x = torch.nn.functional.conv2d(x, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(x, sobel_y, padding=1)

        # è®¡ç®—è¾¹ç•Œæ€»å˜å·®
        btv_loss = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))

        return btv_loss
    

class WeaklySupervisedSegmentationLoss(nn.Module):
    def __init__(self, args,pos_weight=None):
        """
        åˆå§‹åŒ–å¬±ç›‘ç£åˆ†åˆ¹æŸå¤±å‡½æ•°
        :param pos_weight: æ­£æ ·æœ¬æƒé‡ï¼Œç”¨äºå¤„ç†ç±»åˆ«ä¸å¹³è¡¡
        """
        super(WeaklySupervisedSegmentationLoss, self).__init__()
        self.tv_loss=TotalVariationLoss()
        self.btv_loss=BoundaryTotalVariationLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.dice_loss = DiceLoss()
        self.use_tv=args.use_tv_loss
        self.use_btv=args.use_btv_loss
        self.tv_lambda=args.tv_lambda
        self.btv_lambda=args.btv_lambda

    def forward(self, logits, targets):
        """
        è®¡ç®—å¼³ç›‘ç åˆ†å‰²å‡½æ•°
        :param logits: æ¨¡å‹çš„è¾“å‡ºï¼Œå½¢çŠ¶ä¸º [N, H, W]
        :param targets: æ ‡ç­¾å›¾ï¼Œå€¼ä¸º {255: ç›®æ ‡å½“æ™ºç¤°æŠ±ï¼Œ127: èƒŒæ™¯åˆ’ç—•ï¼Œ0: æœªæ ‡æ³¨åŒºåŸŸ}ï¼Œå½¢çŠ¶ä¸º [N, H, W]
        :return: è®¡ç®—å‡ºçš„å¹³å‡æ‘Ÿå¤±
        """
        import logging
        logger = logging.getLogger('train_debug')
        
        # ç›‘æ§è¾“å…¥
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error(f"[LossFunc debug] logits contains NaN/Inf before loss calculation")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logger.error(f"[LossFunc debug] targets contains NaN/Inf")
        
        # æ„å»ºæ©ç¨ï¼Œæ’é™¤æœªæ–‡æ³¨åŒºåŸŸ
        # ä½¿ç”¨èŒff5å›¥æµ‹ä½ï¼šæ­æ–‡æ³¨åŒºåŸŸæ¥è¿šni0{ï#9bcy¦kÏŒ{ï#: ã9¦kùg*ŒËLù.búeíˆX\ÚÈH
\™Ù]ÈˆŒJK™›Ø]

HÈ9§*¹¨!ù¬ê9c.¹gçù£©z/äL;ï#9am¹/fy..ŒBˆÈ9l!¹bcy¦kÊŒJz/k9£h¹..ŒKŒ;ï#: ã9¦kÊŒËLÊz/k9£h¹..ŒŒˆš[˜\Wİ\™Ù]ÈH
\™Ù]ÈˆJK™›Ø]

B‚ˆÈ9b!¹b*ú+¨yë¥ğÑy£kùi,yd£XÙy£gùi,Bˆ˜ÙWÛÜÜ×İ˜[HÙ[‹˜˜ÙWÛÜÜÊÙÚ]Ëš[˜\Wİ\™Ù]ÊBˆXÙWÛÜÜ×İ˜[HÙ[‹™XÙWÛÜÜÊÙÚ]Ëš[˜\Wİ\™Ù]ÊBˆˆÈ9æäy£©ùd!:`ê9b!¹£gùi,BˆYˆÜ˜Úš\Û˜[Š˜ÙWÛÜÜ×İ˜[
K˜[J
HÜˆÜ˜Úš\Ú[™Š˜ÙWÛÜÜ×İ˜[
K˜[J
N‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×H˜ÙWÛÜÜÈÛÛZ[œÈ˜S‹Ò[™ˆŠBˆYˆÜ˜Úš\Û˜[ŠXÙWÛÜÜ×İ˜[
K˜[J
HÜˆÜ˜Úš\Ú[™ŠXÙWÛÜÜ×İ˜[
K˜[J
N‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×HXÙWÛÜÜÈÛÛZ[œÈ˜S‹Ò[™ˆŠBˆˆÈ9d"9nm¹gîùèk¹£gùi,Bˆ˜\ÙWÛÜÜÈH
˜ÙWÛÜÜ×İ˜[
ÈXÙWÛÜÜ×İ˜[
H
ˆX\ÚÂˆˆÈ9æäy£©ùgîùèk¹£gùi,BˆYˆÜ˜Úš\Û˜[Š˜\ÙWÛÜÜÊK˜[J
HÜˆÜ˜Úš\Ú[™Š˜\ÙWÛÜÜÊK˜[J
N‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×H˜\ÙWÛÜÜÈÛÛZ[œÈ˜S‹Ò[™ˆY\ˆX\ÚÈŠBˆˆÜÜÈH˜\ÙWÛÜÜÂˆˆÈ9/oùå*¹«hùb&yc%‚ˆYˆÙ[‹\ÙWİ‚ˆ—ÛÜÜ×İ˜[HÙ[‹—ÛÜÜÊÙÚ]ÊH
ˆÙ[‹—Û[X™BˆYˆÜ˜Úš\Û˜[Š—ÛÜÜ×İ˜[
K˜[J
HÜˆÜ˜Úš\Ú[™Š—ÛÜÜ×İ˜[
K˜[J
N‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×H—ÛÜÜÈÛÛZ[œÈ˜S‹Ò[™ˆŠBˆÜÜÈHÜÜÈ
È—ÛÜÜ×İ˜[ˆˆÈ:`ê9å*•¹«hùb&yc%‚ˆYˆÙ[‹\ÙWØ‚ˆ—ÛÜÜ×İ˜[HÙ[‹˜—ÛÜÜÊÙÚ]ÊH
ˆÙ[‹˜—Û[X™BˆYˆÜ˜Úš\Û˜[Š—ÛÜÜ×İ˜[
K˜[J
HÜˆÜ˜Úš\Ú[™Š—ÛÜÜ×İ˜[
K˜[J
N‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×H—ÛÜÜÈÛÛZ[œÈ˜S‹Ò[™ˆŠBˆÜÜÈHÜÜÈ
È—ÛÜÜ×İ˜[ˆˆÈ9æäy£©ù§ 9îâ9£gùi,BˆYˆÜ˜Úš\Û˜[ŠÜÜÊK˜[J
HÜˆÜ˜Úš\Ú[™ŠÜÜÊK˜[J
N‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×Hš[˜[ÜÜÈÛÛZ[œÈ˜S‹Ò[™ˆ™Y›Ü™H]™\˜YÚ[™ÈŠBˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×H˜ÙWÛÜÜÎˆØ˜ÙWÛÜÜ×İ˜[›YX[Š
Kš][J
N‹™ŸKXÙWÛÜÜÎˆÙXÙWÛÜÜ×İ˜[›YX[Š
Kš][J
N‹™ŸHŠBˆYˆÙ[‹\ÙWİ‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×H—ÛÜÜÎˆİ—ÛÜÜ×İ˜[›YX[Š
Kš][J
N‹™ŸHŠBˆYˆÙ[‹\ÙWØ‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×H—ÛÜÜÎˆØ—ÛÜÜ×İ˜[›YX[Š
Kš][J
N‹™ŸHŠBˆˆÈ9¨.y£k¹£ªyè z+¨yë¥ù§"y¥b9c.¹gçùæ¡9nlùgaù£gùi,{ï"9¥è9§"y¥b9¨!ù¬ê9¥íº/å9fçˆ;ï#:`oùacHÌ8¡¤ˆ˜S»ï"Bˆ˜[YHX\ÚËœİ[J
BˆYˆ˜[YN‚ˆ™]\›ˆÜÜËœİ[J
H
ˆŒˆˆš[˜[ÛÜÜÈHÜÜËœİ[J
HÈ˜[YˆˆÈ9æäy£©ù§ 9îâ9nlùgaù£gùi,BˆYˆÜ˜Úš\Û˜[Šš[˜[ÛÜÜÊK˜[J
HÜˆÜ˜Úš\Ú[™Šš[˜[ÛÜÜÊK˜[J
N‚ˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×Hš[˜[ÛÜÜÈÛÛZ[œÈ˜S‹Ò[™ˆY\ˆ]™\˜YÚ[™ÈŠBˆÙÙÙ\‹™\œ›ÜŠˆ–ÓÜÜÑ[˜ÈXY×H˜[Y^[Îˆİ˜[Yš][J
_HŠBˆˆ™]\›ˆš[˜[ÛÜÜÂ‚‚ˆˆˆ‚‚‚‚™YˆX\š×Ø\™XJ[YËKÛÛÜ‹˜Y]\ÏMKÚ\OIÜÜ]X\™IÊN‚ˆÈ[YÎˆĞËZ[ˆH[

BˆHH[
JBˆÈH[YËœÚ\VÎŒ—BˆŒˆH˜Y]\È
ˆ˜Y]\Âˆ›Üˆ[ˆ˜[™ÙJ\˜Y]\Ë˜Y]\ÊÌJN‚ˆ›ÜˆH[ˆ˜[™ÙJ\˜Y]\Ë˜Y]\ÊÌJN‚ˆYˆÚ\HOH	ØÚ\˜ÛIÈ[™
™
ÈJ™HˆŒ‚ˆÛÛ[YBˆYˆH
ÙÈ[™HJÙH‚ˆ[YÖŞJÙK
Ù—HHÛÛÜ‚‚‚‚š[\ÜÛÜ›šXK™š[\œÈ\È×Ùš[\œÂ‚ˆÈ9å*:/æy.*¹aïy¥l9¦ïù£h¹/h9æ¡[\KÜØÚ\H9âb9§+™Yˆ[˜ÛÙWÜ›Û\×İ×ÛX\ÙÜJ›Û\ËX™[ËËÚYÛXOLKŒ]šXÙOIØİYIË˜]ÚÜÚ^™OS›Û™JN‚ˆˆˆ‚ˆ9l!œ›Û\ùï%¹è y..¹àëyb¦ùfï‚ˆˆ\™ÜÎ‚ˆ›Û\Îˆ9i ¹§§9¦+Û\İ;ï#9b&y«ãù.*¹a`ùí(9¦+ú+éX˜]Ú9¨-ù§+9æ¡›Û\ùb%ú(jÖŞKLWKŞ‹L—K‹‹—Bˆ9i ¹§§9¦+ùcey.*¹b%ú(j;ï#9b&z)á¹..¹cey¨-ù§+;ï"9d$yd#¹ao9k®{ï"BˆX™[Îˆ9i ¹§§9¦+Û\İ;ï#9b&y«ãù.*¹a`ùí(9¦+ú+éX˜]Ú9¨-ù§+9æ¡X™[ùb%ú(jÛK‹‹‹—Bˆ9i ¹§§9¦+ùcey.*¹b%ú(j;ï#9b&z)á¹..¹cey¨-ù§+;ï"9d$yd#¹ao9k®{ï"BˆÎˆ9àëyb¦ùfï¹l.¹kîˆÚYÛXNˆ:jæ9¥«ùª(yìâœÚYÛXBˆ]šXÙNˆ:+¯¹i!Âˆ˜]ÚÜÚ^™Nˆ˜]Ú9i)ùl#ûï#9i ¹§§9..“›Û™yb&z!ê¹bª9£ª9¥«Bˆˆ™]\›œÎ‚ˆ›Û\ÛX\ˆĞ‹‹×H9¢%ˆÌK‹×{ï"9cey¨-ù§+9¥í»ï"Bˆˆˆ‚ˆÈ9b)9¥«y¦+Ø˜]Ú:/æ9¦+ùcey¨-ù§+;ï"9d$yd#¹ao9k®{ï"BˆÈ9i ¹§§˜]ÚÜÚ^™H9¦#¹èk¹£!ùk¦¹.%ˆ{ï#9/oùå*˜]Ú9ª(yo#ÂˆÈ9¢%º !yi ¹§§›Û\È9¦+ùb%ú(j9æ¡9b%ú(j;ï"˜]Ú9¨/9o#ûï"{ï#9.gù/oùå*˜]Ú9ª(yo#Âˆ\×Ø˜]ÚH˜[ÙBˆYˆ˜]ÚÜÚ^™H\È›İ›Û™H[™˜]ÚÜÚ^™HˆN‚ˆ\×Ø˜]ÚHYBˆ[Yˆ[Š›Û\ÊHˆ‚ˆÈ9¨à9§éy¦+ùd)¹¦+Ø˜]Ú9¨/9o#ûï&¹ë+9. 9.*¹a`ùí(9¦+ùb%ú(j;ï#9.%9ë+9. 9.*¹a`ùí(9æ¡9ë+9. 9.*¹a`ùí(9¦+ùgd9¨!ùkîBˆYˆ\Ú[œİ[˜ÙJ›Û\ÖÌK\İ
N‚ˆYˆ[Š›Û\ÖÌJHˆ[™\Ú[œİ[˜ÙJ›Û\ÖÌVÌK
\İ\JJH[™[Š›Û\ÖÌVÌJHOH‚ˆ\×Ø˜]ÚHYBˆˆYˆ›İ\×Ø˜]Ú‚ˆÈ9cey¨-ù§+9ª(yo#ûï"9d$yd#¹ao9k®{ï"Bˆ›Û\ÛX\HÜ˜Ú™\›ÜÊ
K‹ÊK\O]Ü˜Ú™›Ø]Ì‹]šXÙOY]šXÙJBˆYˆ[Š›Û\ÊHOHÜˆ[ŠX™[ÊHOH‚ˆ™]\›ˆ›Û\ÛX\‚ˆÈ9èk¹/çH›Û\È9d£X™[È:eoùn©¹. :!íˆYˆ[Š›Û\ÊHOH[ŠX™[ÊN‚ˆ™]\›ˆ›Û\ÛX\‚ˆÈ9¨à9§éH›Û\È9.+yæ¡9a`ùí(9¦+ùd)¹¦+ùgd9¨!ùkîy¨/9o#ÂˆN‚ˆ›Üˆ›Û\Ú][KX™[[ˆš\
›Û\ËX™[ÊN‚ˆÈ9i ¹§§›Û\Ú][H9.#y¦+ùgd9¨!ùkîy¨/9o#ûï#:-ìú/áÂˆYˆ›İ\Ú[œİ[˜ÙJ›Û\Ú][K
\İ\JJHÜˆ[Š›Û\Ú][JHOH‚ˆÛÛ[YBˆHH›Û\Ú][VÌK›Û\Ú][VÌWBˆH[
›İ[™

ˆ
ËLJHÈL
JHÈ9`aú+¯ˆ9¦+ÈL9gd9¨!ÂˆHH[
›İ[™
H
ˆ
LJHÈL
JHÈ9`aú+¯ˆH9¦+ÈL9gd9¨!ÂˆYˆHÈ[™HH‚ˆ›Û\ÛX\ÌX™[KHHKŒÈ9`aú+¯ˆX™[9¦+È9¢%ˆBˆ^Ù\
˜[YQ\œ›Ü‹\Q\œ›ÜŠN‚ˆÈ9i ¹§§:)èùc!yi,z-){ï#:/å9fç¹ên¹àëyb¦ùfï‚ˆ™]\›ˆ›Û\ÛX\ˆˆÈ9g*Ôy."º/æú(c:jæ9¥«ùª(yìâ»ï"9cey¨-ù§+9ª(yo#ûï"BˆÙ\›™[ÜÚ^™HH[
ˆ
ˆ›İ[™
ÚYÛXH
ˆÊH
ÈJBˆYˆÙ\›™[ÜÚ^™Hˆ‚ˆ›Û\ÛX\H×Ùš[\œË™Ø]\ÜÚX[—Ø›\Œ™
›Û\ÛX\
Ù\›™[ÜÚ^™KÙ\›™[ÜÚ^™JK
ÚYÛXKÚYÛXJJBˆˆ™]\›ˆ›Û\ÛX\ˆ[ÙN‚ˆÈ˜]Ú9ª(yo#ûï"9d$zaãùc%¹âb9§+;ï#9­¢:fi]Ûˆ›Üˆ9oª¹ã«ûï"BˆYˆ˜]ÚÜÚ^™H\È›Û™N‚ˆ˜]ÚÜÚ^™HH[Š›Û\ÊHYˆ›Û\È[ÙHBˆˆ›Û\ÛX\HÜ˜Ú™\›ÜÊ
˜]ÚÜÚ^™K‹ÊK\O]Ü˜Ú™›Ø]Ì‹]šXÙOY]šXÙJBˆˆÈ9l!¹¢`9§"H˜]Ú9æ¡›Û\È9d£X™[È9leynlù..ˆ[œÛÜˆ9¤ãy/gˆ[ØˆH×Bˆ[ŞH×Bˆ[ŞHH×Bˆ[ÛX™[H×Bˆ›Üˆˆ[ˆ˜[™ÙJ˜]ÚÜÚ^™JN‚ˆYˆˆ[Š›Û\ÊH[™[Š›Û\ÖØ—JHˆ‚ˆ—ÛX™[ÈHX™[ÖØ—HYˆ
ˆ[ŠX™[ÊH[™[ŠX™[ÖØ—JHˆ
H[ÙH×BˆˆHZ[Š[Š›Û\ÖØ—JK[Š—ÛX™[ÊJBˆ›ÜˆY[ˆ˜[™ÙJŠN‚ˆ›Û\Ú][HH›Û\ÖØ—VÚYBˆYˆ\Ú[œİ[˜ÙJ›Û\Ú][K
\İ\JJH[™[Š›Û\Ú][JHOH‚ˆ[Ø‹˜\[™
ŠBˆ[Ş˜\[™
›Û\Ú][VÌJBˆ[ŞK˜\[™
›Û\Ú][VÌWJBˆ[ÛX™[˜\[™
[
—ÛX™[ÖÚYJJBˆˆYˆ[Š[ØŠHˆ‚ˆ—ÚYHÜ˜Ú[œÛÜŠ[Ø‹\O]Ü˜Ú›Û™Ë]šXÙOY]šXÙJBˆØÛÛÜ™ÈHÜ˜Ú[œÛÜŠ[Ş\O]Ü˜Ú™›Ø]Ì‹]šXÙOY]šXÙJBˆWØÛÛÜ™ÈHÜ˜Ú[œÛÜŠ[ŞK\O]Ü˜Ú™›Ø]Ì‹]šXÙOY]šXÙJBˆÚYHÜ˜Ú[œÛÜŠ[ÛX™[\O]Ü˜Ú›Û™Ë]šXÙOY]šXÙJBˆˆÛX\YH
ØÛÛÜ™È
ˆ
ÈHJHÈL
Kœ›İ[™

K›Û™Ê
BˆWÛX\YH
WØÛÛÜ™È
ˆ
HJHÈL
Kœ›İ[™

K›Û™Ê
Bˆˆ˜[YH
ÛX\YH
H	ˆ
ÛX\YÊH	ˆ
WÛX\YH
H	ˆ
WÛX\Y
BˆYˆ˜[Y˜[J
N‚ˆ›Û\ÛX\Ø—ÚYİ˜[YKÚYİ˜[YKWÛX\Yİ˜[YKÛX\Yİ˜[YWHHKŒˆˆÈ9g*Ôy."º/æú(c:jæ9¥«ùª(yìâ»ï"˜]Ú9i!9ä!»ï"BˆÙ\›™[ÜÚ^™HH[
ˆ
ˆ›İ[™
ÚYÛXH
ˆÊH
ÈJBˆYˆÙ\›™[ÜÚ^™Hˆ‚ˆ›Û\ÛX\H×Ùš[\œË™Ø]\ÜÚX[—Ø›\Œ™
›Û\ÛX\
Ù\›™[ÜÚ^™KÙ\›™[ÜÚ^™JK
ÚYÛXKÚYÛXJJBˆˆ™]\›ˆ›Û\ÛX\