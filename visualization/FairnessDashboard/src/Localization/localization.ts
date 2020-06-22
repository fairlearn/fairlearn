import LocalizedStrings from 'localized-strings';
import cs from './cs-CZ.json';
import de from './de-DE.json';
import en from './en.json';
import es from './es-ES.json';
import fr from './fr-FR.json';
import hu from './hu-HU.json';
import it from './it-IT.json';
import ja from './ja-JP.json';
import ko from './ko-KR.json';
import nl from './nl-NL.json';
import pl from './pl-PL.json';
import ptbr from './pt-BR.json';
import pt from './pt-PT.json';
import ru from './ru-RU.json';
import sv from './sv-SE.json';
import tr from './tr-TR.json';
import zhcn from './zh-CN.json';
import zhtw from './zh-TW.json';

export const localization = new LocalizedStrings({
    en,
    cs,
    de,
    es,
    fr,
    hu,
    it,
    ja,
    ko,
    nl,
    pl,
    'pt-BR': ptbr,
    'pt-PT': pt,
    ru,
    sv,
    tr,
    'zh-CN': zhcn,
    'zh-TW': zhtw,
}) as any;
